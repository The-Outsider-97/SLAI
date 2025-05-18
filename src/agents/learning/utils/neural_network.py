
import torch
import yaml
import math

CONFIG_PATH = "src/agents/learning/configs/learning_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        # Merge nested dictionaries recursively
        for key in user_config:
            if key in base_config and isinstance(base_config[key], dict) and isinstance(user_config[key], dict):
                base_config[key].update(user_config[key])
            else:
                base_config[key] = user_config[key]
    return base_config

# --- Activation Functions ---
# Based on common activation functions in deep learning literature.
class Activation:
    """Base class for activation functions."""
    def forward(self, z):
        """Computes the activation."""
        raise NotImplementedError
    def backward(self, z_or_a_depending_on_context):
        """Computes the derivative of the activation function w.r.t. its input z."""
        raise NotImplementedError

class ReLU(Activation):
    """Rectified Linear Unit activation.
    Reference: Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines.
    """
    def forward(self, z):
        return torch.maximum(torch.tensor(0.0, device=z.device, dtype=z.dtype), z)
    def backward(self, z):
        return (z > 0).type_as(z)

class Sigmoid(Activation):
    """Sigmoid activation function."""
    def forward(self, z):
        return 1 / (1 + torch.exp(-z))
    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)

class Tanh(Activation):
    """Hyperbolic Tangent (Tanh) activation function."""
    def forward(self, z):
        return torch.tanh(z)
    def backward(self, z):
        t = self.forward(z) # or torch.tanh(z)
        return 1 - t**2

class Linear(Activation):
    """Linear activation function (identity)."""
    def forward(self, z):
        return z
    def backward(self, z):
        return torch.ones_like(z)

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

class NeuralNetwork:
    """Simple 3-layer neural network with manual backpropagation"""
    
    def __init__(self, config, input_dim, output_dim, hidden_dim=128):
        # He initialization with ReLU
        self.W1 = torch.randn(input_dim, hidden_dim) * torch.sqrt(torch.tensor(2. / input_dim))
        self.b1 = torch.zeros(hidden_dim)
        self.W2 = torch.randn(hidden_dim, hidden_dim) * torch.sqrt(torch.tensor(2. / hidden_dim))
        self.b2 = torch.zeros(hidden_dim)
        self.W3 = torch.randn(hidden_dim, output_dim) * torch.sqrt(torch.tensor(2. / hidden_dim))
        self.b3 = torch.zeros(output_dim)

        nn_config = config.get('neural_network', {})
        self.layer_dims = nn_config.get('layer_dims', [input_dim, 128, 64, output_dim])
        self.hidden_activation = nn_config.get('hidden_activation', 'relu')
        self.output_activation = nn_config.get('output_activation', 'linear')
        self.loss_function = nn_config.get('loss_function', 'mse')
        self.optimizer = nn_config.get('optimizer', 'adam')
        self.learning_rate = nn_config.get('learning_rate', 0.001)
        self.layer_dims = nn_config['layer_dims']
        self.num_layers = len(self.layer_dims) - 1 # Number of layers with weights/biases
        self.l1_lambda = nn_config.get('l1_lambda', 0.0)
        self.l2_lambda = nn_config.get('l2_lambda', 0.0)

        self._init_activation_functions(config)
        self._init_loss_function(config)
        self._initialize_weights() # Depends on activation types for He/Xavier
        self._init_optimizer(config)

        # Intermediate values for backprop
        self._cache = {}

    def _str_to_activation(self, name_str):
        name_lower = name_str.lower()
        if name_lower == 'relu': return ReLU()
        if name_lower == 'sigmoid': return Sigmoid()
        if name_lower == 'tanh': return Tanh()
        if name_lower == 'linear': return Linear()
        if name_lower == 'softmax': return Softmax()
        raise ValueError(f"Unknown activation function: {name_str}")

    def _init_activation_functions(self, config):
        hidden_act_str = config.get('hidden_activation', 'relu')
        output_act_str = config.get('output_activation', 'linear')

        self.hidden_activations = []
        if self.num_layers > 1: # If there are hidden layers
            self.hidden_activations = [self._str_to_activation(hidden_act_str) for _ in range(self.num_layers - 1)]
        self.output_activation = self._str_to_activation(output_act_str)

    def _init_loss_function(self, config):
        nn_config = config.get('neural_network', {})  # Access neural_network subsection
        loss_str = nn_config.get('loss_function', 'mse').lower()  # Get loss from nn_config
    
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
            
    def _init_optimizer(self, config):
        optimizer_name = config.get('optimizer', 'sgd').lower()
        lr = config.get('learning_rate', 0.01)

        if optimizer_name == 'sgd':
            self.optimizer = SGD(learning_rate=lr)
        elif optimizer_name == 'momentum':
            beta = config.get('momentum_beta', 0.9)
            self.optimizer = SGDMomentum(learning_rate=lr, beta=beta)
        elif optimizer_name == 'adam':
            beta1 = config.get('adam_beta1', 0.9)
            beta2 = config.get('adam_beta2', 0.999)
            epsilon = config.get('adam_epsilon', 1e-8)
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
            z = current_a @ W + b # Linear transformation

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
        Args:
            X_batch: Input data for the batch.
            y_batch: True labels for the batch.
        Returns:
            The loss value for the batch (scalar).
        """
        # 1. Forward pass to get predictions
        y_pred_output = self.forward(X_batch)
        
        # 2. Compute loss
        # y_pred_output is the activated output. compute_loss handles if logits are needed.
        loss = self.compute_loss(y_pred_output, y_batch)
        
        # 3. Backward pass to compute gradients (dL/dWs, dL/dBs)
        # y_batch is passed for loss computation & format consistency.
        # Gradients are calculated based on cached values and y_batch.
        self.backward(y_batch)
        
        # 4. Update parameters using optimizer
        self.update_parameters()
        
        return loss.item() # Return scalar loss value

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

    config = load_config()
    input_dim = 64
    output_dim = 10

    network = NeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=128,
        config=config)
    
    print(f"\n{network}\n")
    print("\n=== Successfully Neural Network ===\n")

# ====================== Usage Example 2 ======================
if __name__ == "__main__":
    print("\n * * * * Phase 2 * * * *\n=== Neural Network Demonstration ===\n")
    
    # 1. Load configuration
    user_config = {
        'neural_network': {
            'loss_function': 'cross_entropy'  # Only update this key, keep others from base config
        }
    }
    config = get_merged_config(user_config)
    
    # 2. Create synthetic dataset
    input_dim = 784  # MNIST-like input dimension
    output_dim = 10  # 10-class classification
    num_samples = 1000
    X = torch.randn(num_samples, input_dim)  # Random "images"
    y = torch.randint(0, output_dim, (num_samples,))  # Random class labels

    # 3. Initialize network from config
    network = NeuralNetwork(
        config=config,
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    print(f"Initialized network with architecture: {network.layer_dims}")
    print(f"Using {network.hidden_activation} activation | {network.output_activation} output")
    print(f"Optimizer: {network.optimizer} | Learning rate: {network.learning_rate}\n")

    # 4. Training loop
    num_epochs = 50
    batch_size = 32
    losses = []

    print("Training network...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            loss = network.train_step(X_batch, y_batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / (num_samples/batch_size)
        losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    # 5. Evaluation
    with torch.no_grad():
        test_X = torch.randn(100, input_dim)  # Test batch
        test_y = torch.randint(0, output_dim, (100,))
        
        preds = network.predict(test_X)
        predicted_classes = torch.argmax(preds, dim=1)
        accuracy = (predicted_classes == test_y).float().mean()
        
        print(f"\nTest Accuracy: {accuracy.item()*100:.1f}%")

    # 6. Checkpoint demonstration
    checkpoint_path = "src/agents/learning/cache/network_checkpoint.pt"
    print(f"\nSaving model to {checkpoint_path}")
    torch.save(network.get_weights(), checkpoint_path)

    # 7. Load checkpoint verification
    loaded_weights = torch.load(checkpoint_path)
    network.set_weights(loaded_weights)
    
    # Verify matching predictions
    original_pred = network.predict(test_X[0:1])
    network.set_weights(loaded_weights)
    reloaded_pred = network.predict(test_X[0:1])
    
    print("\nCheckpoint verification:", 
          "Match" if torch.allclose(original_pred, reloaded_pred) else "Mismatch")

    print("\n=== Demonstration Complete ===\n")
