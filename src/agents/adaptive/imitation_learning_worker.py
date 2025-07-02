
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Callable, Optional

from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Imitation Learning")
printer = PrettyPrinter

class ImitationLearningWorker:
    """
    Imitation Learning Worker with Behavior Cloning and DAgger
    - Learns from demonstration data
    - Supports online aggregation of expert demonstrations
    - Integrates with RL policies through mixed objectives
    
    Features:
    - Pure behavior cloning for initial training
    - DAgger for ongoing interactive learning
    - Uncertainty-aware sampling for expert queries
    - Seamless integration with RL policies
    """
    def __init__(self, action_dim: int, state_dim: int, policy_network: nn.Module):
        """
        Initialize Imitation Learning Worker
        
        Args:
            action_dim: Dimensionality of action space
            state_dim: Dimensionality of state space
            policy_network: Reference to the policy network to train
        """
        self.config = load_global_config()
        self.batch_size = self.config.get('batch_size')

        self.il_config = get_config_section('imitation_learning')
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.policy_net = policy_network
        
        # Learning parameters
        self.lr = self.il_config.get('learning_rate', 0.001)
        self.clip_value = self.il_config.get('grad_clip', 1.0)
        self.mix_ratio = self.il_config.get('rl_mix_ratio', 0.7)
        self.entropy_threshold = self.il_config.get('entropy_threshold', 0.5)

        self.memory = MultiModalMemory()
        
        # Optimization
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.lr
        )
        self.loss_fn = nn.MSELoss() if self.il_config.get('continuous_actions', False) \
                       else nn.CrossEntropyLoss()
        
        # Demonstration memory
        self.demo_memory = deque(maxlen=self.il_config.get('demo_capacity', 10000))
        self.dagger_memory = deque(maxlen=self.il_config.get('dagger_capacity', 5000))
        
        # DAgger parameters
        self.dagger_frequency = self.il_config.get('dagger_frequency', 5)
        self.query_prob = self.il_config.get('initial_query_prob', 0.8)
        self.query_decay = self.il_config.get('query_decay', 0.99)
        
        # Expert interface
        self.expert_policy = None
        self.update_count = 0
        
        logger.info("Imitation Learning Worker initialized")

    def get_action(self, state: np.ndarray) -> int:
        """Get action from imitation learning policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            output = self.policy_net(state_tensor)
            if self.il_config.get('continuous_actions', False):
                return output.squeeze().numpy()
            else:
                return torch.argmax(output, dim=-1).item()
    
    def register_expert(self, expert: Callable[[np.ndarray], np.ndarray]):
        """Register expert policy function"""
        self.expert_policy = expert
        logger.info("Expert policy registered")
    
    def load_demonstrations(self, demonstrations: List[Dict]):
        """
        Load offline demonstration dataset
        
        Args:
            demonstrations: List of dicts with keys:
                'state': np.ndarray
                'action': np.ndarray or int
        """
        for demo in demonstrations:
            self.demo_memory.append((
                torch.FloatTensor(demo['state']),
                torch.FloatTensor(demo['action']) if self.il_config.get('continuous_actions') 
                else torch.tensor(demo['action'])
            ))
        logger.info(f"Loaded {len(demonstrations)} offline demonstrations")
    
    def add_demonstration(self, state: np.ndarray, action: np.ndarray):
        """Add single demonstration sample"""
        self.memory.store_experience(
            state=state,
            action=action,
            reward=2.0,  # Positive reward for demonstrations
            next_state=None,
            done=True,
            context={"source": "demonstration"}
        )

        self.demo_memory.append((
            torch.FloatTensor(state),
            torch.FloatTensor(action) if self.il_config.get('continuous_actions') 
            else torch.tensor(action)
        ))
    
    def behavior_cloning(self, epochs: int = 10):
        """Train using pure behavior cloning"""
        if len(self.demo_memory) < self.batch_size:
            logger.warning("Insufficient demonstrations for BC training")
            return
        
        samples = self.memory.sample(min(len(self.memory), self.batch_size))
            
        dataset = list(self.demo_memory)
        logger.info(f"Starting Behavior Cloning with {len(dataset)} samples")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(len(dataset))
            
            for i in range(0, len(dataset), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch = [dataset[idx] for idx in batch_indices]
                states, actions = zip(*batch)
                
                states = torch.stack(states)
                actions = torch.stack(actions)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.policy_net(states)
                
                # Calculate loss
                if self.il_config.get('continuous_actions'):
                    loss = self.loss_fn(outputs, actions)
                else:
                    loss = F.cross_entropy(outputs, actions)
                
                # Backpropagate
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(dataset) / self.batch_size)
            logger.info(f"BC Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    def dagger_query(self, state: np.ndarray, agent_action: np.ndarray) -> np.ndarray:
        """
        DAgger interaction - decide whether to query expert and update memory
        
        Returns:
            action: Either expert action or agent action
        """
        if self.expert_policy is None:
            return agent_action
            
        # Decide whether to query expert
        should_query = np.random.rand() < self.query_prob
        
        if should_query:
            expert_action = self.expert_policy(state)
            self.dagger_memory.append((
                torch.FloatTensor(state),
                torch.FloatTensor(expert_action) if self.il_config.get('continuous_actions') 
                else torch.tensor(expert_action)
            ))
            self.query_prob *= self.query_decay  # Reduce query probability
            return expert_action
        
        return agent_action
    
    def uncertainty_query(self, state: np.ndarray) -> bool:
        """
        Uncertainty-based expert query decision
        Uses policy entropy to determine when to query expert
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = F.softmax(self.policy_net(state_tensor), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).item()
            
        return entropy > self.entropy_threshold
    
    def dagger_update(self):
        """Perform DAgger update using aggregated demonstrations"""
        if len(self.dagger_memory) < self.batch_size:
            return
            
        # Combine demo and DAgger memories
        combined_data = list(self.demo_memory) + list(self.dagger_memory)
        indices = np.random.permutation(len(combined_data))
        batch_indices = indices[:self.batch_size]
        batch = [combined_data[idx] for idx in batch_indices]
        
        states, actions = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        
        # Training step
        self.optimizer.zero_grad()
        outputs = self.policy_net(states)
        
        if self.il_config.get('continuous_actions'):
            loss = self.loss_fn(outputs, actions)
        else:
            loss = F.cross_entropy(outputs, actions)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.optimizer.step()
        
        logger.info(f"DAgger Update | Loss: {loss.item():.4f} | Samples: {len(combined_data)}")
        self.update_count += 1
    
    def mixed_objective_update(self, states: torch.Tensor, actions: torch.Tensor, 
                              advantages: torch.Tensor, rl_loss: torch.Tensor):
        """
        Combined RL and imitation learning update
        
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Advantage estimates
            rl_loss: Computed RL policy loss
        """
        # Imitation loss
        policy_output = self.policy_net(states)
        
        if self.il_config.get('continuous_actions'):
            il_loss = self.loss_fn(policy_output, actions)
        else:
            il_loss = F.cross_entropy(policy_output, actions)
        
        # Combined loss
        combined_loss = (self.mix_ratio * rl_loss) + ((1 - self.mix_ratio) * il_loss)
        
        # Update
        self.optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.optimizer.step()
        
        return combined_loss.item()
    
    def get_demonstration_count(self) -> Dict[str, int]:
        """Get current demonstration statistics"""
        return {
            'offline_demos': len(self.demo_memory),
            'dagger_demos': len(self.dagger_memory),
            'total_demos': len(self.demo_memory) + len(self.dagger_memory)
        }
    
    def save_demonstrations(self, filepath: str):
        """Save demonstrations to file"""
        # Implementation depends on storage format
        pass
    
    def load_demonstrations(self, filepath: str):
        """Load demonstrations from file"""
        # Implementation depends on storage format
        pass
