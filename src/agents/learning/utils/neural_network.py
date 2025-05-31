
import torch
import torch.nn as nn
import time

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.activation_engine import Activation, ReLU, Sigmoid, Tanh, Linear
from logs.logger import get_logger

logger = get_logger("Neural Network")

class Softmax(Activation):
    """Softmax activation, typically for multi-class classification output.
    Note: Derivative is complex (Jacobian). Usually combined with CrossEntropyLoss
    for a simpler and more stable gradient (dL/dz = prediction_probs - true_labels).
    """
    def forward(self, z):
        # Subtract max for numerical stability (log-sum-exp trick part 1)
        exp_z = torch.exp(z - torch.max(z, dim=-1, keepdim=True)[0])
        return exp_z / torch.sum(exp_z, dim=-1, keepdim=True)
    
    def backward(self, z):
        # The derivative of Softmax (da/dz) is a Jacobian matrix.
        # dL/dz = dL/da * da/dz.
        # This is complex and rarely computed directly in backprop.
        # Instead, dL/dz for (Softmax + CrossEntropy) is much simpler.
        # If this layer is used and its derivative is needed directly (e.g. MSE after Softmax),
        # this method would need to compute dL/da_output @ Jacobian_softmax.
        # For now, we assume it's either the final output not needing derivative or handled by CrossEntropyLoss.
        raise NotImplementedError("Softmax derivative is usually handled within SoftmaxCrossEntropyLoss. Set output_activation to 'linear' if using CrossEntropyLoss.")

# --- Loss Functions ---
class Loss:
    """Base class for loss functions."""
    def forward(self, y_pred, y_true):
        """Computes the loss."""
        raise NotImplementedError
    def backward(self, y_pred, y_true, batch_size):
        """Computes the gradient of the loss w.r.t. y_pred."""
        raise NotImplementedError

class MSELoss(Loss):
    """Mean Squared Error Loss.
    L = (1/N) * sum((y_pred - y_true)^2)
    dL/dy_pred = (2/N) * (y_pred - y_true)
    """
    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true)**2)
    def backward(self, y_pred, y_true, batch_size):
        if batch_size == 0: batch_size = 1 # Avoid division by zero
        return (2 / batch_size) * (y_pred - y_true)

class CrossEntropyLoss(Loss):
    """Cross-Entropy Loss with integrated Softmax.
    Expects raw logits as y_pred and class indices as y_true.
    L = - (1/N) * sum(log(softmax(logits)_correct_class))
    dL/dlogits = (1/N) * (softmax(logits) - y_true_one_hot)
    """
    def __init__(self):
        self.softmax = Softmax()
        self._cache = {} # To store probabilities from forward pass

    def forward(self, logits, y_true_indices):
        # logits: (batch_size, num_classes)
        # y_true_indices: (batch_size,) tensor of class indices
        batch_size = logits.shape[0]
        
        # Compute softmax probabilities
        probs = self.softmax.forward(logits)
        self._cache['probs'] = probs
        
        # Select the probabilities of the true classes
        # Add epsilon for numerical stability to prevent log(0)
        y_true_indices = y_true_indices.long()
        log_probs = torch.log(probs[torch.arange(batch_size), y_true_indices] + 1e-9)
        
        # Compute mean negative log likelihood
        loss = -torch.mean(log_probs)
        return loss

    def backward(self, logits, y_true_indices, batch_size):
        # Gradient of CrossEntropyLoss w.r.t. logits (inputs to Softmax)
        # is (probs - y_true_one_hot) / batch_size
        if batch_size == 0: batch_size = 1 # Avoid division by zero
        
        probs = self._cache.get('probs')
        # If backward is called without forward (e.g. direct testing), recompute probs
        if probs is None or probs.shape[0] != logits.shape[0]: 
            probs = self.softmax.forward(logits)

        num_classes = logits.shape[1]
        
        # Create one-hot encoded y_true
        y_true_one_hot = torch.zeros_like(probs)
        y_true_one_hot[torch.arange(batch_size), y_true_indices] = 1.0
        
        # Gradient dL/dlogits
        grad = (probs - y_true_one_hot) / batch_size
        return grad

# --- Optimizers ---
# Based on common optimization algorithms in deep learning.
class Optimizer:
    """Base class for optimizers."""
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        """Updates parameters based on gradients."""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        for i in range(len(params_Ws)):
            params_Ws[i] -= self.learning_rate * grads_dWs[i]
            params_bs[i] -= self.learning_rate * grads_dBs[i]

class SGDMomentum(Optimizer):
    """SGD with Momentum.
    Reference: Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013).
               On the importance of initialization and momentum in deep learning.
    v_new = beta * v_old + grad
    param_new = param_old - lr * v_new
    """
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_Ws = None
        self.v_bs = None

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        if self.v_Ws is None: # Initialize velocities on first step
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]

        for i in range(len(params_Ws)):
            # Update velocities
            self.v_Ws[i] = self.beta * self.v_Ws[i] + grads_dWs[i]
            self.v_bs[i] = self.beta * self.v_bs[i] + grads_dBs[i]
            
            # Update parameters
            params_Ws[i] -= self.learning_rate * self.v_Ws[i]
            params_bs[i] -= self.learning_rate * self.v_bs[i]

class Adam(Optimizer):
    """Adam Optimizer.
    Reference: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_Ws, self.v_Ws = None, None # First and second moment estimates for Ws
        self.m_bs, self.v_bs = None, None # First and second moment estimates for bs
        self.t = 0 # Timestep counter

    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        self.t += 1
        if self.m_Ws is None: # Initialize moment estimates on first step
            self.m_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.m_bs = [torch.zeros_like(b) for b in params_bs]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]

        for i in range(len(params_Ws)):
            # Update biased first moment estimate for Ws
            self.m_Ws[i] = self.beta1 * self.m_Ws[i] + (1 - self.beta1) * grads_dWs[i]
            # Update biased second raw moment estimate for Ws
            self.v_Ws[i] = self.beta2 * self.v_Ws[i] + (1 - self.beta2) * (grads_dWs[i]**2)
            
            # Compute bias-corrected first moment estimate for Ws
            m_hat_W = self.m_Ws[i] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate for Ws
            v_hat_W = self.v_Ws[i] / (1 - self.beta2**self.t)
            
            # Update Ws
            params_Ws[i] -= self.learning_rate * m_hat_W / (torch.sqrt(v_hat_W) + self.epsilon)

            # Update biased first moment estimate for bs
            self.m_bs[i] = self.beta1 * self.m_bs[i] + (1 - self.beta1) * grads_dBs[i]
            # Update biased second raw moment estimate for bs
            self.v_bs[i] = self.beta2 * self.v_bs[i] + (1 - self.beta2) * (grads_dBs[i]**2)

            # Compute bias-corrected first moment estimate for bs
            m_hat_b = self.m_bs[i] / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate for bs
            v_hat_b = self.v_bs[i] / (1 - self.beta2**self.t)

            # Update bs
            params_bs[i] -= self.learning_rate * m_hat_b / (torch.sqrt(v_hat_b) + self.epsilon)

class NeuralNetwork(torch.nn.Module):
    """Simple 3-layer neural network with manual backpropagation"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # He initialization with ReLU
        # self.W1 = torch.randn(input_dim, hidden_dim) * torch.sqrt(torch.tensor(2. / input_dim))
        # self.b1 = torch.zeros(hidden_dim)
        # self.W2 = torch.randn(hidden_dim, hidden_dim) * torch.sqrt(torch.tensor(2. / hidden_dim))
        # self.b2 = torch.zeros(hidden_dim)
        # self.W3 = torch.randn(hidden_dim, output_dim) * torch.sqrt(torch.tensor(2. / hidden_dim))
        # self.b3 = torch.zeros(output_dim)
        self.config = load_global_config()
        self.nn_config = get_config_section('neural_network')
        self.layer_dims = self.nn_config.get('layer_dims', [input_dim, 128, 64, output_dim])

        # Override input/output dims explicitly to ensure compatibility
        self.layer_dims[0] = input_dim
        self.layer_dims[-1] = output_dim

        self.hidden_activation = self.nn_config.get('hidden_activation', 'relu')
        self.output_activation = self.nn_config.get('output_activation', 'linear')
        self.loss_function = self.nn_config.get('loss_function', 'mse')
        self.optimizer = self.nn_config.get('optimizer', 'adam')
        self.learning_rate = self.nn_config.get('learning_rate', 0.001)
        self.num_layers = len(self.layer_dims) - 1 # Number of layers with weights/biases
        self.l1_lambda = self.nn_config.get('l1_lambda', 0.0)
        self.l2_lambda = self.nn_config.get('l2_lambda', 0.0)

        self._init_activation_functions()
        self._init_loss_function()
        self._initialize_weights() # Depends on activation types for He/Xavier
        self._init_optimizer()

        # Intermediate values for backprop
        self._cache = {}

        logger.info(f"Learning Neural Network succesfully initialized")

    def _str_to_activation(self, name_str):
        name_lower = name_str.lower()
        if name_lower == 'relu': return ReLU()
        if name_lower == 'sigmoid': return Sigmoid()
        if name_lower == 'tanh': return Tanh()
        if name_lower == 'linear': return Linear()
        if name_lower == 'softmax': return Softmax()
        raise ValueError(f"Unknown activation function: {name_str}")

    def _init_activation_functions(self):
        hidden_act_str = self.nn_config.get('hidden_activation', 'relu')
        output_act_str = self.nn_config.get('output_activation', 'linear')

        self.hidden_activations = []
        if self.num_layers > 1: # If there are hidden layers
            self.hidden_activations = [self._str_to_activation(hidden_act_str) for _ in range(self.num_layers - 1)]
        self.output_activation = self._str_to_activation(output_act_str)

    def _init_loss_function(self):
        loss_str = self.nn_config.get('loss_function', 'mse').lower()
    
        if loss_str == 'mse':
            self.loss_fn = MSELoss()
        elif loss_str == 'cross_entropy':
            if not isinstance(self.output_activation, Linear):
                print("Warning: CrossEntropyLoss expects logits (linear output activation).")
            self.loss_fn = CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_str}")

    def _initialize_weights(self):
        """
        Initializes weights and biases.
        Uses He initialization for ReLU, Xavier/Glorot for Sigmoid/Tanh/Softmax.
        Refs:
        - He et al. (2015). "Delving Deep into Rectifiers..." (He initialization)
        - Glorot & Bengio (2010). "Understanding the difficulty of training deep feedforward neural networks" (Xavier/Glorot initialization)
        """
        self.Ws = []
        self.bs = []
        for i in range(self.num_layers):
            fan_in = self.layer_dims[i]
            fan_out = self.layer_dims[i+1]

            # Determine activation for current layer to choose init strategy
            if i < self.num_layers - 1: # Hidden layer
                current_activation_type = self.hidden_activations[i]
            else: # Output layer
                current_activation_type = self.output_activation
            
            if isinstance(current_activation_type, ReLU):
                # He initialization: std = sqrt(2 / fan_in)
                std_dev = torch.sqrt(torch.tensor(2.0 / fan_in))
            elif isinstance(current_activation_type, (Sigmoid, Tanh, Softmax, Linear)): # Linear also often uses Xavier
                # Xavier/Glorot initialization: std = sqrt(1 / fan_in) or sqrt(2 / (fan_in + fan_out))
                # Using sqrt(1 / fan_in) variant here.
                std_dev = torch.sqrt(torch.tensor(1.0 / fan_in))
            else: # Fallback for unknown activation types (should not happen with current setup)
                std_dev = torch.sqrt(torch.tensor(1.0 / fan_in)) 

            self.Ws.append(torch.randn(fan_in, fan_out) * std_dev)
            self.bs.append(torch.zeros(fan_out))
            
    def _init_optimizer(self):
        optimizer_name = self.nn_config.get('optimizer', 'sgd').lower()
        lr = self.nn_config.get('learning_rate', 0.01)

        if optimizer_name == 'sgd':
            self.optimizer = SGD(learning_rate=lr)
        elif optimizer_name == 'momentum':
            beta = self.nn_config.get('momentum_beta', 0.9)
            self.optimizer = SGDMomentum(learning_rate=lr, beta=beta)
        elif optimizer_name == 'adam':
            beta1 = self.nn_config.get('adam_beta1', 0.9)
            beta2 = self.nn_config.get('adam_beta2', 0.999)
            epsilon = self.nn_config.get('adam_epsilon', 1e-8)
            self.optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def forward(self, X):
        """Performs a forward pass through the network."""
        self._cache['inputs'] = X # Input to the first layer
        self._cache['layer_outputs'] = [] # List to store {'z': ..., 'a': ...} for each layer

        current_a = X # Activation from previous layer (or input X)
        for i in range(self.num_layers):
            W, b = self.Ws[i], self.bs[i]
            z = current_a @ W + b  # Linear transformation
            current_a = activation(z)

            if i < self.num_layers - 1: # Hidden layer
                current_a = self.hidden_activations[i].forward(z)
            else: # Output layer
                current_a = self.output_activation.forward(z)
            
            self._cache['layer_outputs'].append({'z': z, 'a': current_a})
        
        return current_a # Final output of the network

    def compute_loss(self, y_pred_output, y_true):
        """
        Computes the loss.
        Args:
            y_pred_output: The final activated output from the network's forward pass.
            y_true: True labels. Format depends on the loss function
                    (e.g., class indices for CrossEntropyLoss, continuous values for MSELoss).
        Returns:
            Scalar tensor representing the loss.
        """
        if isinstance(self.loss_fn, CrossEntropyLoss):
            # CrossEntropyLoss expects logits (pre-softmax).
            # The 'z' of the final layer is these logits.
            final_layer_z = self._cache['layer_outputs'][-1]['z']
            return self.loss_fn.forward(final_layer_z, y_true)
        else:
            # Other losses (like MSE) expect the activated output.
            return self.loss_fn.forward(y_pred_output, y_true)

    def backward(self, y_true):
        """
        Performs backpropagation to compute gradients for weights and biases.
        Relies on values stored in self._cache from the forward pass.
        Args:
            y_true: True labels.
        """
        # Determine batch size (m)
        # y_true shape can be (batch_size,) for CE or (batch_size, output_dim) for MSE
        m = y_true.shape[0] if y_true.dim() > 0 else 1
        if m == 0: m = 1 # Avoid division by zero, though batch size shouldn't be 0

        # Initialize gradients for Ws and bs as lists of zero tensors
        self.dWs = [torch.zeros_like(W, device=W.device) for W in self.Ws]
        self.dBs = [torch.zeros_like(b, device=b.device) for b in self.bs]

        # --- Initial delta (gradient w.r.t. z of the output layer, dL/dz_L) ---
        final_layer_cache = self._cache['layer_outputs'][-1] # Contains z_L and a_L
        
        if isinstance(self.loss_fn, CrossEntropyLoss):
            # For CrossEntropyLoss, backward method returns dL/d(logits) where logits = z_L
            # y_true are class indices for CrossEntropyLoss
            delta = self.loss_fn.backward(final_layer_cache['z'], y_true, m)
        else:
            # For other losses (e.g., MSE), loss_fn.backward returns dL/da_L
            # We need dL/dz_L = dL/da_L * da_L/dz_L
            # y_true matches shape of a_L for MSELoss
            dL_daL = self.loss_fn.backward(final_layer_cache['a'], y_true, m)
            g_prime_zL = self.output_activation.backward(final_layer_cache['z'])
            delta = dL_daL * g_prime_zL

        # --- Backpropagate delta through layers (from L to 1) ---
        for i in reversed(range(self.num_layers)):
            # a_prev is the input to the current layer i (i.e., activation from layer i-1)
            # If i=0 (first layer), a_prev is the network input X
            a_prev = self._cache['inputs'] if i == 0 else self._cache['layer_outputs'][i-1]['a']

            # Gradient w.r.t. weights W_i: dL/dW_i = a_{i-1}.T @ delta_i
            self.dWs[i] = a_prev.T @ delta
            # Gradient w.r.t. biases b_i: dL/db_i = sum(delta_i, axis=0)
            self.dBs[i] = torch.sum(delta, axis=0)

            # Add regularization gradients (if applicable)
            # L2 regularization: d(0.5 * lambda * W^2)/dW = lambda * W
            # Here, loss is often (1/m) * sum(...) + (lambda / (2*m)) * sum(W^2)
            # So dW_reg = (lambda / m) * W
            if self.l2_lambda > 0:
                self.dWs[i] += (self.l2_lambda / m) * self.Ws[i] # L2 penalty applied to weights
            # L1 regularization: d(lambda * |W|)/dW = lambda * sign(W)
            # So dW_reg = (lambda / m) * sign(W)
            if self.l1_lambda > 0:
                self.dWs[i] += (self.l1_lambda / m) * torch.sign(self.Ws[i]) # L1 penalty
            
            if i > 0: # If not the first layer, propagate delta to the previous layer
                # Calculate dL/da_{i-1} = delta_i @ W_i.T
                da_prev = delta @ self.Ws[i].T
                # Get g'(z_{i-1}) for the previous layer's activation
                g_prime_z_prev = self.hidden_activations[i-1].backward(self._cache['layer_outputs'][i-1]['z'])
                # Update delta for the previous layer: delta_{i-1} = dL/da_{i-1} * g'(z_{i-1})
                delta = da_prev * g_prime_z_prev
                
    def update_parameters(self):
        """Updates network parameters using the configured optimizer and computed gradients."""
        self.optimizer.step(self.Ws, self.bs, self.dWs, self.dBs)

    def train_step(self, X_batch, y_batch):
        """
        Performs a single training step: forward pass, loss computation,
        backward pass (gradient computation), and parameter update.
        
        Expanded to include:
        - Input validation
        - Gradient clipping
        - NaN value checks
        - Detailed logging
        - Performance timing
        
        Args:
            X_batch: Input data for the batch (Tensor of shape [batch_size, input_dim])
            y_batch: True labels for the batch (Tensor shape depends on loss function)
            
        Returns:
            The loss value for the batch (scalar float)
            
        Raises:
            ValueError: If input dimensions are invalid
            RuntimeError: If NaNs are detected in gradients or outputs
        """
        # 1. Validate inputs
        if X_batch.dim() != 2:
            raise ValueError(f"X_batch must be 2D tensor, got {X_batch.dim()}D")
        if X_batch.shape[0] == 0:
            raise ValueError("Batch size cannot be zero")
        if X_batch.shape[1] != self.layer_dims[0]:
            raise ValueError(f"Input feature dimension mismatch. Expected {self.layer_dims[0]}, got {X_batch.shape[1]}")
        
        # Start timing for performance monitoring
        forward_time = time.time()
        
        try:
            # 2. Forward pass to get predictions
            y_pred_output = self.forward(X_batch)
            
            # Check for NaN in outputs
            if torch.isnan(y_pred_output).any():
                raise RuntimeError("NaN detected in network output during forward pass")
                
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            logger.debug(f"Input shape: {X_batch.shape}, Input range: [{X_batch.min()}, {X_batch.max()}]")
            raise
            
        forward_time = time.time() - forward_time
        
        loss_time = time.time()
        
        try:
            # 3. Compute loss
            loss = self.compute_loss(y_pred_output, y_batch)
            
            # Check for invalid loss
            if torch.isnan(loss):
                raise RuntimeError("Loss is NaN")
            if torch.isinf(loss):
                raise RuntimeError("Loss is infinite")
                
        except Exception as e:
            logger.error(f"Loss computation failed: {str(e)}")
            logger.debug(f"y_true range: [{y_batch.min()}, {y_batch.max()}]")
            logger.debug(f"y_pred range: [{y_pred_output.min()}, {y_pred_output.max()}]")
            raise
            
        loss_time = time.time() - loss_time
        
        backward_time = time.time()
        
        try:
            # 4. Backward pass to compute gradients
            self.backward(y_batch)
            
            # 5. Gradient clipping to prevent explosion
            max_grad_norm = 5.0  # Configurable value
            for i in range(len(self.dWs)):
                # Weight gradients
                grad_norm = torch.norm(self.dWs[i])
                if grad_norm > max_grad_norm:
                    self.dWs[i] = self.dWs[i] * (max_grad_norm / (grad_norm + 1e-6))
                    
                # Bias gradients
                grad_norm = torch.norm(self.dBs[i])
                if grad_norm > max_grad_norm:
                    self.dBs[i] = self.dBs[i] * (max_grad_norm / (grad_norm + 1e-6))
                    
                # Check for NaN in gradients
                if torch.isnan(self.dWs[i]).any() or torch.isnan(self.dBs[i]).any():
                    raise RuntimeError(f"NaN detected in gradients at layer {i}")
                    
        except Exception as e:
            logger.error(f"Backward pass failed: {str(e)}")
            logger.debug(f"Loss value: {loss.item()}")
            raise
            
        backward_time = time.time() - backward_time
        
        update_time = time.time()
        
        try:
            # 6. Update parameters using optimizer
            self.update_parameters()
        except Exception as e:
            logger.error(f"Parameter update failed: {str(e)}")
            raise
            
        update_time = time.time() - update_time
        
        # Log performance metrics
        import logging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Train step timings: "
                f"Forward: {forward_time:.4f}s, "
                f"Loss: {loss_time:.4f}s, "
                f"Backward: {backward_time:.4f}s, "
                f"Update: {update_time:.4f}s, "
                f"Total: {forward_time+loss_time+backward_time+update_time:.4f}s"
            )
            
            # Log gradient norms
            for i, (dw, db) in enumerate(zip(self.dWs, self.dBs)):
                logger.debug(
                    f"Layer {i} gradient norms: "
                    f"Weights: {torch.norm(dw):.6f}, "
                    f"Biases: {torch.norm(db):.6f}"
                )
        
        return loss.item()

    def predict(self, X):
        """
        Makes predictions for input X. Essentially a forward pass.
        For classification with CrossEntropyLoss, this returns raw logits if output_activation is linear.
        If probabilities are needed, apply Softmax explicitly or set output_activation to 'softmax'.
        """
        return self.forward(X)

    def get_weights(self):
        """Returns the network's weights and biases."""
        return {'Ws': [W.clone().detach() for W in self.Ws], 
                'bs': [b.clone().detach() for b in self.bs]}

    def set_weights(self, weights_dict):
        """
        Sets the network's weights and biases.
        Args:
            weights_dict (dict): A dictionary with keys 'Ws' and 'bs',
                                 containing lists of tensors for weights and biases.
        """
        if 'Ws' not in weights_dict or 'bs' not in weights_dict:
            raise ValueError("weights_dict must contain 'Ws' and 'bs' keys.")
        if len(weights_dict['Ws']) != self.num_layers or len(weights_dict['bs']) != self.num_layers:
            raise ValueError("Mismatch in the number of layers for weights/biases.")

        self.Ws = [W.clone().detach() for W in weights_dict['Ws']]
        self.bs = [b.clone().detach() for b in weights_dict['bs']]


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Neural Network ===\n")
    import argparse
    config = load_global_config()
    input_dim = 64
    output_dim = 10

    network = NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim)
    
    print(f"\n{network}\n")
    print("\n=== Successfully Neural Network ===\n")

# ====================== Usage Example 2 ======================
    print("\n * * * * Phase 2 * * * *\n=== Neural Network Demonstration ===\n")

    parser = argparse.ArgumentParser(description='Neural Network Examples')
    parser.add_argument('--example', type=int, default=1, choices=[1, 2],
                        help='Example to run (1: simple init, 2: training demo)')
    args = parser.parse_args()

    if args.example == 1:
        print("\n=== Running Neural Network Example 1 ===\n")
        input_dim = 64
        output_dim = 10

        network = NeuralNetwork(input_dim, output_dim)
        
        # Print configuration details
        print("Network Configuration:")
        print(f"  Layer dimensions: {network.layer_dims}")
        # Handle case where there might be no hidden activations
        hidden_act = 'None'
        if network.hidden_activations:
            hidden_act = type(network.hidden_activations[0]).__name__
        print(f"  Hidden activation: {hidden_act}")
        print(f"  Output activation: {type(network.output_activation).__name__}")
        print(f"  Loss function: {type(network.loss_fn).__name__}")
        print(f"  Optimizer: {type(network.optimizer).__name__}")
        print(f"  Learning rate: {network.learning_rate}")
        print(f"  L1 Lambda: {network.l1_lambda}")
        print(f"  L2 Lambda: {network.l2_lambda}")
        
        print("\n=== Example 1 Complete ===\n")

    else:
        print("\n=== Running Neural Network Example 2 ===\n")
        
        # Create synthetic dataset
        input_dim = 784  # MNIST-like input dimension
        output_dim = 10  # 10-class classification
        num_samples = 1000
        X = torch.randn(num_samples, input_dim)  # Random "images"
        y = torch.randint(0, output_dim, (num_samples,))  # Random class labels

        # Initialize network
        network = NeuralNetwork(input_dim, output_dim)
        
        # Print configuration details
        print("Network Configuration:")
        # Handle case where there might be no hidden activations
        hidden_act = 'None'
        if network.hidden_activations:
            hidden_act = type(network.hidden_activations[0]).__name__
        print(f"  Layer dimensions: {network.layer_dims}")
        print(f"  Hidden activation: {hidden_act}")
        print(f"  Output activation: {type(network.output_activation).__name__}")
        print(f"  Loss function: {type(network.loss_fn).__name__}")
        print(f"  Optimizer: {type(network.optimizer).__name__}")
        print(f"  Learning rate: {network.learning_rate}")
        print(f"  L1 Lambda: {network.l1_lambda}")
        print(f"  L2 Lambda: {network.l2_lambda}")
        print()

        # Training loop
        num_epochs = 5  # Reduced for demonstration
        batch_size = 32
        losses = []

        print(f"Training network for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                loss = network.train_step(X_batch, y_batch)
                epoch_loss += loss
                num_batches += 1
            
            # Calculate average loss per batch
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

        # Evaluation
        with torch.no_grad():
            test_X = torch.randn(100, input_dim)  # Test batch
            test_y = torch.randint(0, output_dim, (100,))
            
            preds = network.predict(test_X)
            predicted_classes = torch.argmax(preds, dim=1)
            accuracy = (predicted_classes == test_y).float().mean()
            
            print(f"\nTest Accuracy: {accuracy.item()*100:.1f}%")

        # Checkpoint demonstration
        checkpoint_path = "network_checkpoint.pt"
        print(f"\nSaving model to {checkpoint_path}")
        torch.save(network.get_weights(), checkpoint_path)

        # Load checkpoint into a NEW network for verification
        new_network = NeuralNetwork(input_dim, output_dim)
        loaded_weights = torch.load(checkpoint_path)
        new_network.set_weights(loaded_weights)
        
        # Verify matching predictions
        test_sample = test_X[0:1]
        original_pred = network.predict(test_sample)
        reloaded_pred = new_network.predict(test_sample)
        torch.allclose(original_pred, reloaded_pred)
        
        print("\nCheckpoint verification:", 
              "Match" if torch.allclose(original_pred, reloaded_pred, atol=1e-6) else "Mismatch")

        print("\n=== Example 2 Complete ===\n")
