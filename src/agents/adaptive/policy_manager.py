import numpy as np
import yaml, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.policy_network import PolicyNetwork
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Parameter Tuner")
printer = PrettyPrinter

class PolicyManager:
    def __init__(self):
        self.config = load_global_config()
        self.manager_config = get_config_section('policy_manager')
        self.hidden_layers = self.manager_config.get('hidden_layers')
        self.activation = self.manager_config.get('activation')
        
        self.rl_config = get_config_section('rl')
        self.state_dim = self.rl_config.get('state_dim')
        self.action_dim = self.rl_config.get('action_dim')

        self.memory = MultiModalMemory()
        self.policy_network = PolicyNetwork(
            state_size=self.state_dim,
            action_size=self.action_dim
        )

        self.weights = None # self._initialize_weights()
        self.network = None # self._build_network()
        self._steps = 0
        self.sample = 0

        # Policy performance tracking
        self.param_history = []
        logger.info("Policy Manager initialized with:")
        logger.info(f"State Dim: {self.state_dim}, Action Dim: {self.action_dim}")

    def _initialize_weights(self) -> dict:
        """Initialize policy network weights using Xavier initialization"""
        printer.status("INIT", "Weights succesfully initialized")

        return {
            'input': nn.init.xavier_uniform_(torch.empty(self.state_dim, 64)),
            'hidden': nn.init.xavier_uniform_(torch.empty(64, 32)),
            'output': nn.init.xavier_uniform_(torch.empty(32, self.action_dim))
        }

    def _build_network(self) -> nn.Sequential:
        """Create policy network architecture"""
        printer.status("INIT", "Network builder succesfully initialized")

        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def attach_policy_network(self, policy_network):
        """
        Attach an external PolicyNetwork instance.
        """
        printer.status("INIT", "Policy attacher succesfully initialized")

        self.policy_network = policy_network
        state_size = self.state_dim 
        action_size = self.action_dim
        self.config = self.manager_config
        policy_network = PolicyNetwork(
            state_size=state_size,
            action_size=action_size)
        logger.info("External PolicyNetwork attached to PolicyManager.")
    
    def compute_policy(self, state: np.ndarray) -> np.ndarray:
        printer.status("INIT", "Policy compute succesfully initialized")

        state_tensor = torch.FloatTensor(state).unsqueeze(0) if state.ndim == 1 else torch.FloatTensor(state)
        with torch.no_grad():
            if self.policy_network:
                probs = self.policy_network.forward(state_tensor)
            else:
                probs = self.network(state_tensor)
        return probs.squeeze(0).numpy()

    def update_policy(self, states, actions, advantages):
        """
        Update policy using collected experiences
        - states: Batch of states
        - actions: Batch of actions taken
        - advantages: Computed advantages for each transition
        """
        rewards = [exp['reward'] for exp in self.memory.episodic]
        advantages = self.advantage_calculation(rewards)

        # Convert to tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        advantages_t = torch.FloatTensor(advantages)
        
        # Compute policy gradients
        probs = self.policy_network(states_t)
        log_probs = torch.log(probs.gather(1, actions_t.unsqueeze(1)))
        loss = -(log_probs * advantages_t).mean()
        
        # Backpropagate and update
        self.policy_network.backward(loss)
        self.policy_network.update_parameters()
        
        # Analyze parameter impact periodically
        if len(self.param_history) % 100 == 0:
            impact = self.memory.analyze_parameter_impact()
            logger.debug(f"Parameter impacts: {impact}")

    def update(self, state: np.ndarray, action: int, td_error: float, learning_rate: float = 0.01):
        """Single-sample update method (for compatibility)"""
        # Validate state dimensions
        if len(state) != self.state_dim:
            logger.warning(f"State dimension mismatch in update: expected {self.state_dim}, got {len(state)}")
            state = state[:self.state_dim]

        if self.policy_network:
            # Construct dL/dlogits assuming Softmax output and REINFORCE-style gradient
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.policy_network.forward(state_tensor)
            grad_logits = torch.zeros_like(probs)
            grad_logits[0, action] = 1.0
            grad_logits -= probs
            grad_logits *= td_error
    
            self.policy_network.backward(grad_logits)
            self.policy_network.update_parameters()
        else:
            # fallback to manual update
            state_tensor = torch.FloatTensor(state).requires_grad_(True)
            probs = self.network(state_tensor)
            log_prob = torch.log(probs[action])
            log_prob.backward()
            with torch.no_grad():
                for param in self.network.parameters():
                    param -= learning_rate * td_error * param.grad
                self.network.zero_grad()

    def advantage_calculation(self, rewards, gamma=0.99, normalize=True):
        """
        Calculate advantages from a list of rewards using discounted return.
    
        Args:
            rewards (list or np.ndarray): List of rewards for each step in an episode.
            gamma (float): Discount factor.
            normalize (bool): Whether to normalize advantages to zero mean and unit variance.
    
        Returns:
            np.ndarray: Array of advantages (discounted returns).
        """
        printer.status("CALC", "Starting advantage calculation")
    
        R = 0
        advantages = []
    
        # Calculate discounted returns in reverse
        for r in reversed(rewards):
            R = r + gamma * R
            advantages.insert(0, R)
    
        advantages = np.array(advantages)
    
        if normalize:
            mean = advantages.mean()
            std = advantages.std() + 0.0000001
            advantages = (advantages - mean) / std
    
        printer.status("CALC", "Advantage calculation completed")
        return advantages

    def _step_count(self):
        return self._steps

    def get_action(self, state, explore=True, context=None):
        """Get action with state validation and memory integration"""
        # Validate state dimensions
        if len(state) != self.state_dim:
            logger.warning(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
            state = state[:self.state_dim]  # Truncate to expected dimensions
        
        # Convert state to tensor
        state_t = torch.FloatTensor(state).unsqueeze(0)

        # Retrieve relevant memories
        memory_context = context or {"state": state}
        memories = self.memory.retrieve(
            query="policy_decision",
            context=memory_context
        )
        
        # Use policy network for primary decision
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.policy_network(state_t).numpy()[0]
        
        # Exploration adjustment based on memory
        if explore and memories:
            memory_bias = self.memory._generate_memory_bias(memories)
            adjusted_probs = self._adjust_probs(probs, memory_bias)
            return np.random.choice(self.action_dim, p=adjusted_probs)

        return np.argmax(probs) if not explore else np.random.choice(self.action_dim, p=probs)
    
    def _adjust_probs(self, probs: np.ndarray, memory_bias: np.ndarray) -> np.ndarray:
        """
        Adjust policy probabilities using memory-based bias.
        Applies multiplicative adjustment to action probabilities using:
            adjusted_probs = probs * exp(memory_bias)
        Followed by normalization to ensure valid probability distribution.
        
        Args:
            probs: Original action probabilities from policy network
            memory_bias: Bias vector from memory system
            
        Returns:
            Adjusted probability distribution over actions
        """
        # Ensure valid probability distribution
        if not np.isclose(np.sum(probs), 1.0) or np.any(probs < 0):
            logger.warning("Invalid probability distribution in bias adjustment")
            return probs
        
        # Apply exponential scaling to create multiplicative factor
        adjustment_factor = np.exp(memory_bias)
        
        # Apply adjustment
        adjusted_probs = probs * adjustment_factor
        
        # Normalize to valid probability distribution
        total = np.sum(adjusted_probs)
        if total < 1e-8:  # Avoid division by zero
            logger.warning("Adjusted probabilities sum to zero, using uniform distribution")
            return np.ones_like(probs) / len(probs)
        
        adjusted_probs /= total
        return adjusted_probs

    def save_weights(self, path: str) -> None:
        """Save policy weights to file"""
        torch.save({
            'network_state': self.network.state_dict(),
            'weights': self.weights
        }, path)

    def load_weights(self, path: str) -> None:
        """Load policy weights from file"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state'])
        self.weights = checkpoint['weights']

    def store_experience(self, state, action, reward, 
        next_state=None, done=False, context=None, params=None):
        
        experience = self.memory.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            context=context,
            params=params
        )
        
        if params:
            # Create properly formatted parameters for logging
            log_params = {
                'learning_rate': params.get('learning_rate', np.nan),  # Fixed key
                'exploration_rate': params.get('exploration_rate', np.nan),
                'discount_factor': params.get('discount_factor', np.nan),
                'temperature': params.get('temperature', np.nan)
            }
            self.memory.log_parameters(reward, log_params)

    def consolidate_memory(self):
        """Apply memory forgetting mechanisms"""
        self.memory.consolidate()

    def save_checkpoint(self, path):
        """Save policy and memory state"""
        torch.save({
            'policy_state': self.policy_network.get_weights(),
            'memory_state': self.memory.state_dict()
        }, path)

    def load_checkpoint(self, path):
        """Load full agent state"""
        checkpoint = torch.load(path)
        self.policy_network.set_weights(checkpoint['policy_state'])
        self.memory.load_state_dict(checkpoint['memory_state'])

if __name__ == "__main__":
    print("\n=== Running Policy Manager ===\n")
    printer.status("TEST", "Starting Policy Manager tests", "info")

    manager = PolicyManager()
    print(manager)

    print("\n* * * * * Phase 2 * * * * *\n")
    print("\nAll tests completed successfully!\n")
