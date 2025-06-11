
import torch
import torch.nn as nn
import yaml, json
import torch.optim as optim

from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.activation_engine import Activation, ReLU, Sigmoid, Tanh, Linear
from logs.logger import get_logger

logger = get_logger("Policy Network")

class Softmax(Activation):
    """Softmax activation."""
    def forward(self, z):
        exp_z = torch.exp(z - torch.max(z, dim=-1, keepdim=True)[0]) # Stability
        return exp_z / torch.sum(exp_z, dim=-1, keepdim=True)
    
    def backward(self, z):
        # The derivative da/dz of Softmax is a Jacobian.
        # dL/dz = dL/da * (da/dz).
        # However, for policy gradients, often dL/dz (gradient w.r.t. logits) is computed directly by the RL algorithm.
        # If dL/da is provided, this derivative is needed.
        # S = self.forward(z) # (batch, num_classes)
        # For a single vector s = softmax(z):
        # d s_i / d z_j = s_i * (delta_ij - s_j)
        # This means dS/dz is a batch of Jacobian matrices.
        # S_diag = torch.diag_embed(S) # (batch, num_classes, num_classes)
        # S_outer = S.unsqueeze(-1) @ S.unsqueeze(-2) # (batch, num_classes, num_classes)
        # jacobian = S_diag - S_outer
        # If this function is called, it should return this Jacobian or be used in context of dL/da @ Jacobian.
        # For simplicity in policy network backprop, we assume the agent provides dL/dz_L or the product dL/da_L * (da_L/dz_L).
        # Here, we'll return the component g'(z) such that dL/dz = dL/da * g'(z).
        # For Softmax, this is more complex than element-wise multiplication.
        # This backward is typically used if the loss is (e.g.) MSE *after* Softmax.
        # For typical policy gradient (e.g. REINFORCE), the gradient calculation is simpler w.r.t. logits.
        # Here, we expect the gradient calculation `dL_daL * self.output_activation.backward(final_layer_z)`
        # to correctly yield `dL_dzL`. If `dL_daL` is a vector, and `backward(z)` returns a vector,
        # it implies an element-wise product, which is not generally correct for Softmax's Jacobian.
        # *Correction*: For PolicyNetwork's `backward` method, if we pass `dL_daL`,
        # then `delta = dL_daL * g'(zL)`. This works for element-wise activations.
        # For Softmax, if the RL agent provides `dL_daL`, the PolicyNetwork `backward` would need to handle the Jacobian.
        # A common simplification is for the RL agent to compute `dL_dzL` (gradient w.r.t. logits) directly.
        # If we stick to `dL_daL * g'(zL)` pattern, then `g'(zL)` for softmax should be `1` and the RL agent provides `dL_dlogits * (something)`
        # OR the RL agent computes `dL_dprobs` and then policy network must handle `dprobs/dlogits`.
        # Given typical PG algorithms, it's often `grad_log_pi`. The `log_pi` involves `logits`.
        # For now, assuming the provided `dL_daL` to `PolicyNetwork.backward()` can be element-wise multiplied with
        # `self.output_activation.backward(z_final)` to get `dL_dz_final`.
        # This is true for `Linear`, `ReLU`, `Tanh`, `Sigmoid`. For `Softmax`, this interpretation
        # means `self.output_activation.backward(z)` should effectively be the part of the Jacobian that dL/daL multiplies.
        # This is usually implicitly handled by the RL loss formulation (e.g., `(probs - target_one_hot)` is `dL/dlogits`).
        # Let's assume `backward(z)` for Softmax returns `1` for now, implying `dL/dz = dL/da`, and the agent's `dL/da`
        # is already the "effective" gradient that works like `dL/dlogits`.
        # This is a common simplification when `dL_daL` is effectively `dL_dlogits` adjusted for advantage.
        return torch.ones_like(z) # Placeholder: assumes agent computes dL/dlogits effectively.

# --- Optimizers (Copied/adapted from neural_network.py for self-containment) ---
class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        for i in range(len(params_Ws)):
            params_Ws[i] -= self.learning_rate * grads_dWs[i]
            params_bs[i] -= self.learning_rate * grads_dBs[i]

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_Ws = None
        self.v_bs = None
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        if self.v_Ws is None:
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]
        for i in range(len(params_Ws)):
            self.v_Ws[i] = self.beta * self.v_Ws[i] + grads_dWs[i]
            self.v_bs[i] = self.beta * self.v_bs[i] + grads_dBs[i]
            params_Ws[i] -= self.learning_rate * self.v_Ws[i]
            params_bs[i] -= self.learning_rate * self.v_bs[i]

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_Ws, self.v_Ws = None, None
        self.m_bs, self.v_bs = None, None
        self.t = 0
    def step(self, params_Ws, params_bs, grads_dWs, grads_dBs):
        self.t += 1
        if self.m_Ws is None:
            self.m_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.v_Ws = [torch.zeros_like(W) for W in params_Ws]
            self.m_bs = [torch.zeros_like(b) for b in params_bs]
            self.v_bs = [torch.zeros_like(b) for b in params_bs]
        
        # new_params_Ws = []
        # new_params_bs = []
        for i in range(len(params_Ws)):
            # Update moment estimates
            self.m_Ws[i] = self.beta1 * self.m_Ws[i] + (1 - self.beta1) * grads_dWs[i]
            self.v_Ws[i] = self.beta2 * self.v_Ws[i] + (1 - self.beta2) * grads_dWs[i]**2
            m_hat_W = self.m_Ws[i] / (1 - self.beta1**self.t)
            v_hat_W = self.v_Ws[i] / (1 - self.beta2**self.t)
    
            self.m_bs[i] = self.beta1 * self.m_bs[i] + (1 - self.beta1) * grads_dBs[i]
            self.v_bs[i] = self.beta2 * self.v_bs[i] + (1 - self.beta2) * grads_dBs[i]**2
            m_hat_b = self.m_bs[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_bs[i] / (1 - self.beta2**self.t)
    
            # Compute new values and update using .data.copy_
            updated_W = params_Ws[i] - self.learning_rate * m_hat_W / (torch.sqrt(v_hat_W) + self.epsilon)
            updated_b = params_bs[i] - self.learning_rate * m_hat_b / (torch.sqrt(v_hat_b) + self.epsilon)
    
            params_Ws[i].data.copy_(updated_W)
            params_bs[i].data.copy_(updated_b)    

class NoveltyDetector(nn.Module):
    """
    Simple neural network for estimating state novelty
    """
    def __init__(self, input_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.target = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)

    def forward(self, x):
        with torch.no_grad():
            target_features = self.target(x)
        pred_features = self.predictor(x)
        return torch.norm(pred_features - target_features, dim=1)

class PolicyNetwork(torch.nn.Module):
    """
    Manually implemented Policy Network for Reinforcement Learning.
    Outputs action probabilities (e.g., via Softmax) or action parameters (e.g., via Tanh/Linear).
    Backpropagation is driven by gradients from an RL algorithm (e.g., REINFORCE, A2C, PPO).
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.config = load_global_config()
        self.pn_config = get_config_section('policy_network')
        if not self.pn_config:
            self.pn_config = self.config.get('neural_network', {})
            logger.info("Using neural_network config for PolicyNetwork")
        
        if not self.pn_config:
            logger.warning("PolicyNetwork configuration not found. Using defaults.")    

        hidden_layer_sizes = self.pn_config.get('hidden_layer_sizes', [128, 64]) # Default architecture
        self.layer_dims = [state_size] + hidden_layer_sizes + [action_size]
        self.num_layers = len(self.layer_dims) - 1

        self.l1_lambda = self.pn_config.get('l1_lambda', 0.0)
        self.l2_lambda = self.pn_config.get('l2_lambda', 0.0)

        self._init_activation_functions()
        self._initialize_weights()
        self._init_optimizer()

        self._cache = {}

        logger.info("PolicyNetwork succesfully initialized.")

    def _str_to_activation(self, name_str):
        name_lower = name_str.lower()
        if name_lower == 'relu': return ReLU()
        if name_lower == 'sigmoid': return Sigmoid()
        if name_lower == 'tanh': return Tanh()
        if name_lower == 'linear': return Linear()
        if name_lower == 'softmax': return Softmax()
        raise ValueError(f"Unknown activation function: {name_str}")

    def _init_activation_functions(self):
        hidden_act_str = self.pn_config.get('hidden_activation', 'relu')
        output_act_str = self.pn_config.get('output_activation', 'softmax') # Softmax common for discrete policies

        self.hidden_activations = []
        if self.num_layers > 1: # If there are hidden layers
            self.hidden_activations = [self._str_to_activation(hidden_act_str) for _ in range(self.num_layers - 1)]
        self.output_activation = self._str_to_activation(output_act_str)

    def _initialize_weights(self):
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        for i in range(self.num_layers):
            fan_in = self.layer_dims[i]
            fan_out = self.layer_dims[i+1]

            current_activation_type = self.hidden_activations[i] if i < self.num_layers - 1 else self.output_activation
            
            if isinstance(current_activation_type, ReLU):
                std_dev = torch.sqrt(torch.tensor(2.0 / fan_in)) # He initialization
            elif isinstance(current_activation_type, (Sigmoid, Tanh, Softmax, Linear)):
                std_dev = torch.sqrt(torch.tensor(1.0 / fan_in)) # Xavier/Glorot initialization (variant)
            else:
                std_dev = torch.sqrt(torch.tensor(1.0 / fan_in)) 

            self.Ws.append(nn.Parameter(torch.randn(fan_in, fan_out) * std_dev))
            self.bs.append(nn.Parameter(torch.zeros(fan_out)))

    def _init_optimizer(self):
        optimizer_cfg = self.config.get('optimizer_config', {})
        optimizer_name = optimizer_cfg.get('type', 'adam').lower()
        lr = optimizer_cfg.get('learning_rate', 0.001)

        if optimizer_name == 'sgd':
            self.optimizer = SGD(learning_rate=lr)
        elif optimizer_name == 'momentum':
            beta = optimizer_cfg.get('momentum_beta', 0.9)
            self.optimizer = SGDMomentum(learning_rate=lr, beta=beta)
        elif optimizer_name == 'adam':
            beta1 = optimizer_cfg.get('adam_beta1', 0.9)
            beta2 = optimizer_cfg.get('adam_beta2', 0.999)
            epsilon = optimizer_cfg.get('adam_epsilon', 1e-8)
            self.optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def forward(self, state):
        """
        Performs a forward pass to compute action probabilities or parameters.
        Returns:
            torch.Tensor: Output of the network (e.g., action probabilities or action parameters).
        """
        self._cache['inputs'] = state
        self._cache['layer_outputs'] = []

        current_a = state
        for i in range(self.num_layers):
            W, b = self.Ws[i], self.bs[i]
            z = current_a @ W + b

            if i < self.num_layers - 1: # Hidden layer
                current_a = self.hidden_activations[i].forward(z)
            else: # Output layer
                current_a = self.output_activation.forward(z)
            
            self._cache['layer_outputs'].append({'z': z, 'a': current_a})
        
        return current_a

    def backward(self, dL_daL):
        """
        Performs backpropagation to compute gradients for weights and biases.
        The RL agent is responsible for computing dL_daL, the gradient of the
        RL objective function with respect to the policy network's activated output (aL).

        Args:
            dL_daL (torch.Tensor): Gradient of the loss w.r.t. the network's final activated output.
                                   Shape should match the network's output shape.
        """
        m = dL_daL.shape[0] if dL_daL.dim() > 0 else 1
        if m == 0: m = 1

        self.dWs = [torch.zeros_like(W) for W in self.Ws]
        self.dBs = [torch.zeros_like(b) for b in self.bs]

        # --- Initial delta (gradient w.r.t. z of the output layer, dL/dz_L) ---
        # dL/dz_L = dL/da_L * da_L/dz_L
        final_layer_cache = self._cache['layer_outputs'][-1]
        g_prime_zL = self.output_activation.backward(final_layer_cache['z']) # da_L/dz_L
        
        # Critical point for Softmax: If self.output_activation is Softmax, its .backward(z) currently returns 1.
        # This implies that dL_daL *is effectively* dL_dzL (or dL_dlogits).
        # This is often the case in policy gradient algorithms where the gradient w.r.t. logits is derived.
        # e.g. REINFORCE: grad log pi(a|s) * A.  grad_logits log softmax(logits) = (one_hot_action - softmax(logits)).
        # If dL_daL is truly dL/d(probabilities), then Softmax.backward would need to handle the Jacobian,
        # or this step would be more complex. We assume here that the RL agent provides a `dL_daL`
        # that, when multiplied by `g_prime_zL` (which is 1 for Softmax here), yields the correct `dL_dzL`.
        delta = dL_daL * g_prime_zL

        # --- Backpropagate delta through layers ---
        for i in reversed(range(self.num_layers)):
            a_prev = self._cache['inputs'] if i == 0 else self._cache['layer_outputs'][i-1]['a']

            self.dWs[i] = a_prev.T @ delta
            self.dBs[i] = torch.sum(delta, axis=0)

            if self.l2_lambda > 0:
                self.dWs[i] += (self.l2_lambda / m) * self.Ws[i]
            if self.l1_lambda > 0:
                self.dWs[i] += (self.l1_lambda / m) * torch.sign(self.Ws[i])
            
            if i > 0: # Propagate delta to the previous layer
                da_prev = delta @ self.Ws[i].T
                g_prime_z_prev = self.hidden_activations[i-1].backward(self._cache['layer_outputs'][i-1]['z'])
                delta = da_prev * g_prime_z_prev
                
    def update_parameters(self):
        """Updates network parameters using the configured optimizer and computed gradients."""
        if not hasattr(self, 'dWs') or not hasattr(self, 'dBs'):
            logger.error("Gradients not computed. Call backward() before update_parameters().")
            return
        self.optimizer.step(self.Ws, self.bs, self.dWs, self.dBs)

    def get_weights(self):
        return {'Ws': [W.clone().detach() for W in self.Ws], 
                'bs': [b.clone().detach() for b in self.bs]}

    def set_weights(self, weights_dict):
        if 'Ws' not in weights_dict or 'bs' not in weights_dict:
            raise ValueError("weights_dict must contain 'Ws' and 'bs' keys.")
        if len(weights_dict['Ws']) != self.num_layers or len(weights_dict['bs']) != self.num_layers:
            raise ValueError(f"Mismatch in layers. Expected {self.num_layers} layers.")
    
        # Update existing parameters instead of reassigning the list
        for i in range(self.num_layers):
            self.Ws[i].data.copy_(weights_dict['Ws'][i])
            self.bs[i].data.copy_(weights_dict['bs'][i])
        
        # Reset optimizer states
        if hasattr(self.optimizer, 't'):  # Adam
            self.optimizer.t = 0
            self.optimizer.m_Ws, self.optimizer.v_Ws = None, None
            self.optimizer.m_bs, self.optimizer.v_bs = None, None
        if hasattr(self.optimizer, 'v_Ws'):  # Momentum
            self.optimizer.v_Ws, self.optimizer.v_bs = None, None
        logger.info("Weights updated. Optimizer states reset.")

# Example Usage (Conceptual - actual use within an RL agent loop)
if __name__ == '__main__':
    # 1. Create a dummy global_config (normally loaded from YAML)
    sample_global_config = {
        'policy_network': {
            'hidden_layer_sizes': [64, 32],
            'hidden_activation': 'relu',
            'output_activation': 'softmax', # For a discrete action space
            'optimizer_config': {
                'type': 'adam',
                'learning_rate': 0.001
            },
            'l2_lambda': 0.0001
        }
    }
    
    # 2. Define state and action sizes
    state_dim = 4  # E.g., CartPole state
    action_dim = 2 # E.g., CartPole actions (left, right)

    # 3. Instantiate the PolicyNetwork
    policy_net = PolicyNetwork(state_size=state_dim, 
                               action_size=action_dim)
    logger.info("PolicyNetwork initialized.")

    # 4. Example forward pass (inference)
    # Create a dummy batch of states
    # Ensure states are FloatTensors. Default is DoubleTensor if created from Python lists/numbers.
    dummy_states = torch.randn(10, state_dim, dtype=torch.float32) # Batch of 10 states
    
    action_probs = policy_net.forward(dummy_states)
    logger.info(f"Sample action probabilities (first state): {action_probs[0]}")
    assert action_probs.shape == (10, action_dim), "Output shape mismatch"
    if isinstance(policy_net.output_activation, Softmax):
         assert torch.allclose(torch.sum(action_probs, dim=1), torch.tensor(1.0)), "Softmax output doesn't sum to 1"

    # 5. Example backward pass and update (simulating an RL agent's update step)
    # In a real RL agent, dL_daL would be calculated based on the chosen RL algorithm (e.g., REINFORCE)
    # For REINFORCE, if output is softmax: dL/daL might be (A * (1 - P(a_chosen))) for chosen action prob, (A * (-P(a_other))) for others
    # Or more commonly, dL/d_logits = (Advantage * (indicator_chosen_action - action_probabilities))
    # Since our Softmax.backward is simplified, let's assume dL_daL is effectively dL_d_logits for Softmax output.
    # Here's a conceptual placeholder for dL_daL:
    # Assume some advantages and chosen actions leading to a gradient for the output layer's *activations*.
    # For simplicity, let's pretend this is a gradient that makes sense after applying g'(z_L).
    # If output is softmax, dL_daL is often (ChosenActionIndicator - Probs) * Advantage / BatchSize
    # This dL_daL *is* dL/d(logits_L) for a softmax output combined with cross-entropy-like loss.
    # So if Softmax.backward returns 1, this dL_daL will be treated as dL/dzL.
    
    # Example: if output is softmax, let's simulate a gradient w.r.t. logits.
    # This means the agent calculated something like (target_distribution - output_probabilities) * advantage
    dummy_dL_daL = torch.randn(10, action_dim, dtype=torch.float32) * 0.1 
    # If output is Softmax, this dummy_dL_daL conceptually represents dL/d_logits.
    # If output is Tanh, this dummy_dL_daL represents dL/d_tanh_output.

    policy_net.backward(dummy_dL_daL)
    logger.info("Backward pass completed. Gradients computed.")
    
    # 6. Update parameters
    policy_net.update_parameters()
    logger.info("Parameters updated.")

    # 7. Get/Set weights example
    weights = policy_net.get_weights()
    logger.info(f"Number of weight tensors: {len(weights['Ws'])}, bias tensors: {len(weights['bs'])}")
    
    # Create a new network and set weights (e.g., for loading a model)
    new_policy_net = PolicyNetwork(state_size=state_dim, 
                                   action_size=action_dim)
    new_policy_net.set_weights(weights)
    logger.info("Weights set to a new network instance.")

    # Verify weights are the same
    new_weights = new_policy_net.get_weights()
    for i in range(len(weights['Ws'])):
        assert torch.equal(weights['Ws'][i], new_weights['Ws'][i]), f"W{i} mismatch after set_weights"
        assert torch.equal(weights['bs'][i], new_weights['bs'][i]), f"b{i} mismatch after set_weights"
    logger.info("Weight get/set verified.")

    logger.info("PolicyNetwork conceptual example finished.")
