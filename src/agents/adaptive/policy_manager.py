import numpy as np
import yaml, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.policy_network import PolicyNetwork
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from logs.logger import get_logger

logger = get_logger("Parameter Tuner")

class PolicyManager:
    def __init__(self, state_dim: int, action_dim: int):
        self.config = load_global_config()
        self.manager_config = get_config_section('policy_manager')
        memory = MultiModalMemory()
        self.memory = memory

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = self.manager_config.get('hidden_layers', [64, 32])
        self.activation = self.manager_config.get('activation', 'tanh')
        self.weights = self._initialize_weights()
        self.network = self._build_network()
        
        logger.info(f"Succesfully initialize Policy Manager with:\n- {self.state_dim}\n- {self.action_dim}")

    def _initialize_weights(self) -> dict:
        """Initialize policy network weights using Xavier initialization"""
        return {
            'input': nn.init.xavier_uniform_(torch.empty(self.state_dim, 64)),
            'hidden': nn.init.xavier_uniform_(torch.empty(64, 32)),
            'output': nn.init.xavier_uniform_(torch.empty(32, self.action_dim))
        }

    def attach_policy_network(self, config, policy_network):
        """
        Attach an external PolicyNetwork instance.
        """
        self.policy_network = policy_network
        state_size = self.state_dim 
        action_size = self.action_dim
        self.config = config
        policy_network = PolicyNetwork(
            state_size=state_size,
            action_size=action_size,
            config=config.get('policy_network', {}))
        logger.info("External PolicyNetwork attached to PolicyManager.")

    def _build_network(self) -> nn.Sequential:
        """Create policy network architecture"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def compute_policy(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0) if state.ndim == 1 else torch.FloatTensor(state)
        with torch.no_grad():
            if self.policy_network:
                probs = self.policy_network.forward(state_tensor)
            else:
                probs = self.network(state_tensor)
        return probs.squeeze(0).numpy()

    def update(self, state: np.ndarray, action: int, td_error: float, learning_rate: float = 0.01):
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

    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        probs = self.compute_policy(state)
        return np.random.choice(self.action_dim, p=probs) if explore else np.argmax(probs)
    
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

if __name__ == "__main__":
    # Test initialization
    state_dim = 4
    action_dim = 2
    state_size = 4
    action_size = 2
    
    # Load configuration instead of using None
    config = load_global_config()  # <-- Fix: Load default config
    
    policy_network = PolicyNetwork(
        config=config,  # Pass the loaded config
        state_size=state_size, 
        action_size=action_size
    )
    pm = PolicyManager(state_dim, action_dim)
    policy_net = PolicyNetwork(
        config=pm.config,  # Use PolicyManager's config
        state_size=4, 
        action_size=2
    )
    pm.attach_policy_network(config, policy_network)
    print("PolicyManager initialized successfully")

    # Test compute_policy
    dummy_state = np.random.rand(state_dim)
    probs = pm.compute_policy(dummy_state)
    assert np.isclose(probs.sum(), 1.0), "Probabilities should sum to 1"
    print(f"Compute_policy test passed. Action probabilities: {probs}")

    # Test get_action
    print("\nAction sampling test (exploration):")
    for _ in range(5):
        print(f"Sampled action: {pm.get_action(dummy_state)}")
    
    print("\nGreedy action (exploitation):")
    print(f"Greedy action: {pm.get_action(dummy_state, explore=False)}")

    # Test weight update
    if pm.policy_network:
        initial_weights = [w.clone() for w in pm.policy_network.get_weights()['Ws']]
    else:
        initial_weights = [p.clone() for p in pm.network.parameters()]
    
    td_error = 0.5
    dummy_action = np.random.randint(action_dim)
    pm.update(dummy_state, dummy_action, td_error, 0.1)
    
    if pm.policy_network:
        updated_weights = [w.clone() for w in pm.policy_network.get_weights()['Ws']]
    else:
        updated_weights = [p.clone() for p in pm.network.parameters()]
    
    weight_changed = any(not torch.equal(i, u) for i, u in zip(initial_weights, updated_weights))
    assert weight_changed, "Weights should change after update"
    print("\nWeight update test passed")

    # Test save/load functionality
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test/test_weights.pth")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save and load
        pm.save_weights(save_path)
        original_probs = pm.compute_policy(dummy_state)
        
        # Modify weights
        for p in pm.network.parameters():
            p.data += torch.randn_like(p.data)
        
        # Load back
        pm.load_weights(save_path)
        loaded_probs = pm.compute_policy(dummy_state)
        
        assert np.allclose(original_probs, loaded_probs), "Saved/Loaded weights mismatch"
        print("\nSave/load test passed")
    print("\nAll tests completed successfully!")
