import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyManager:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = self._initialize_weights()
        self.network = self._build_network()
        
    def _initialize_weights(self) -> dict:
        """Initialize policy network weights using Xavier initialization"""
        return {
            'input': nn.init.xavier_uniform_(torch.empty(self.state_dim, 64)),
            'hidden': nn.init.xavier_uniform_(torch.empty(64, 32)),
            'output': nn.init.xavier_uniform_(torch.empty(32, self.action_dim))
        }
    
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
        """Compute action probabilities for given state"""
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            probs = self.network(state_tensor)
        return probs.numpy()
    
    def update(self, 
             state: np.ndarray, 
             action: int, 
             td_error: float, 
             learning_rate: float = 0.01) -> None:
        """
        Update policy weights using policy gradient theorem
        
        Args:
            state: Current state
            action: Taken action
            td_error: Temporal Difference error
            learning_rate: Learning rate for update
        """
        # Convert to tensor and enable grad
        state_tensor = torch.FloatTensor(state).requires_grad_(True)
        
        # Forward pass
        probs = self.network(state_tensor)
        log_prob = torch.log(probs[action])
        
        # Calculate gradient
        log_prob.backward()
        
        # Manual SGD update
        with torch.no_grad():
            for param in self.network.parameters():
                param -= learning_rate * td_error * param.grad
            self.network.zero_grad()

    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Sample action from policy distribution
        """
        probs = self.compute_policy(state)
        if explore:
            return np.random.choice(self.action_dim, p=probs)
        else:
            return np.argmax(probs)
    
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
