import yaml
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any, List, Dict

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.learning_memory import LearningMemory
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.agents.adaptive.imitation_learning_worker import ImitationLearningWorker
from src.agents.adaptive.meta_learning_worker import MetaLearningWorker
from src.agents.adaptive.utils.neural_network import ActorCriticNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Skill Worker")
printer = PrettyPrinter

@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: torch.Tensor = None

class SkillWorker:
    """
    Reinforcement Learning worker for skill execution
    - Learns primitive actions to accomplish specific tasks
    - Uses A2C (Actor-Critic) for policy updates
    - Receives goals from PolicyManager when goal-conditioned
    """
    _worker_registry = {}
    
    def __init__(self):
        super().__init__()
        self.skill_id = None
        self.name = None
        self.config = None
        self.worker_config = None
        self.state_dim = None
        self.action_dim = None
        self.goal_dim = None
        self.enable_goals = None
        self.current_goal = None
        self.input_dim = None
        self.local_memory = None
        self.learner_memory = None
        self.actor_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.gamma = None
        self.learning_rate = None
        self.entropy_coef = None
        self.value_coef = None
        self.max_grad_norm = None
        self.imitation_learning = None
        self.reward_normalization = None
        self.reward_clip_range = None
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 1e-4

    @classmethod
    def create_worker(cls, skill_id: int, skill_metadata: dict):
        """Factory method to create and register a new worker"""
        worker = cls()
        worker.initialize(skill_id, skill_metadata)
        cls._worker_registry[skill_id] = worker
        return worker

    @classmethod
    def get_worker(cls, skill_id: int):
        """Retrieve worker from registry"""
        return cls._worker_registry.get(skill_id)

    def initialize(self, skill_id: int, skill_metadata: dict):
        """Initialize worker with specific skill parameters"""
        self.skill_id = skill_id
        self.name = skill_metadata.get('name', f'skill_{skill_id}')
        self.config = load_global_config()
        self.worker_config = get_config_section('skill_worker')
        
        # Environment parameters
        self.state_dim = skill_metadata['state_dim']
        self.action_dim = skill_metadata['action_dim']
        self.goal_dim = self.worker_config.get('goal_dim', 0)
        
        # Goal conditioning setup
        self.enable_goals = self.worker_config.get('enable_goals', False)
        self.current_goal = np.zeros(self.goal_dim) if self.enable_goals else None
        self.input_dim = self.state_dim + (self.goal_dim if self.enable_goals else 0)
        
        # Memory systems
        self.local_memory = MultiModalMemory()
        self.learner_memory = LearningMemory()
        
        # Get actor and critic layers with proper output dimensions
        actor_layers = self.worker_config.get('actor_layers', [64, 64])
        critic_layers = self.worker_config.get('critic_layers', [64, 32])
        
        # Ensure actor outputs action_dim and critic outputs 1
        actor_layers = self._adjust_output_layer(actor_layers, self.action_dim)
        critic_layers = self._adjust_output_layer(critic_layers, 1)
        
        # Actor-Critic Network
        self.actor_critic = ActorCriticNetwork(
            state_dim=self.input_dim,
            action_dim=self.action_dim,
            actor_layers=actor_layers,
            critic_layers=critic_layers
        )
        
        # Learning parameters
        self.gamma = self.worker_config.get('discount_factor', 0.99)
        self.learning_rate = self.worker_config.get('learning_rate', 0.001)
        self.entropy_coef = self.worker_config.get('entropy_coef', 0.01)
        self.value_coef = self.worker_config.get('value_coef', 0.5)
        self.max_grad_norm = self.worker_config.get('max_grad_norm', 0.5)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_critic.actor.parameters(), 
            lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.actor_critic.critic.parameters(), 
            lr=self.learning_rate
        )
        
        # Reward processing
        self.reward_normalization = self.worker_config.get('reward_normalization', True)
        self.reward_clip_range = self.worker_config.get('reward_clip_range', (-10.0, 10.0))
        
        logger.info(f"Skill Worker '{self.name}' initialized with:")
        logger.info(f"State Dim: {self.state_dim}, Action Dim: {self.action_dim}")
        logger.info(f"Goal Conditioning: {self.enable_goals}, Goal Dim: {self.goal_dim}")

    def attach_meta_learning(self, meta_worker: MetaLearningWorker):
        """Connect to meta-learning worker"""
        self.meta_learning = meta_worker
        logger.info(f"Meta Learning Worker attached to SkillWorker '{self.name}'")

    def attach_imitation_learning(self, imitation_worker: ImitationLearningWorker):
        """Connect to meta-learning worker"""
        self.imitation_learning = imitation_worker
        logger.info(f"Imitation Learning Worker attached to SkillWorker '{self.name}'")

    def _adjust_output_layer(self, layers: list, output_dim: int) -> list:
        """Ensure the last layer has the correct output dimension"""
        if not layers:
            return [output_dim]
            
        # If last layer is a dictionary
        if isinstance(layers[-1], dict):
            layers[-1] = layers[-1].copy()
            layers[-1]['neurons'] = output_dim
        else:  # If last layer is an integer
            layers[-1] = output_dim
            
        return layers

    def set_goal(self, goal: np.ndarray):
        """Set current goal for goal-conditioned skills"""
        if self.enable_goals and goal.shape[0] == self.goal_dim:
            self.current_goal = goal
            logger.debug(f"Worker {self.skill_id} set goal: {goal[:4]}...")
        elif self.enable_goals:
            logger.warning(f"Invalid goal dimension: expected {self.goal_dim}, got {goal.shape[0]}")

    def _process_state(self, state: np.ndarray) -> np.ndarray:
        """Augment state with goal if goal-conditioned"""
        if self.enable_goals and self.current_goal is not None:
            return np.concatenate([state, self.current_goal])
        return state

    def select_action(self, state: np.ndarray, explore: bool = True) -> tuple:
        """
        Select primitive action for skill execution
        Returns:
            action: Selected primitive action
            log_prob: Log probability of selected action
            entropy: Entropy of action distribution
        """
        try:
            # Use imitation learning 30% of the time during exploration
            if self.imitation_learning and explore and random.random() < 0.3:
                action = self.imitation_learning.get_action(state)
                return action, 0.0, 0.0  # Dummy values for log_prob and entropy
            
            # Original RL action selection
            processed_state = self._process_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
            
            with torch.no_grad():
                actor_output = self.actor_critic.forward_actor(state_tensor)
                dist = torch.distributions.Categorical(logits=actor_output)
                
                if explore:
                    action = dist.sample()
                else:
                    action = torch.argmax(actor_output, dim=-1)
                    
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
            return action.item(), log_prob.item(), entropy.item()
            
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            # Fallback to random action
            return random.randint(0, self.action_dim - 1), 0.0, 0.0

    def store_experience(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        log_prob: float = 0.0,
        entropy: float = 0.0 
    ):
        """Store experience with all necessary components"""
        # Update reward statistics
        self._update_reward_statistics(reward)
        
        # Apply reward normalization
        reward = self._normalize_reward(reward)
        
        # Create transition
        transition = Transition(
            state=torch.FloatTensor(state),
            action=torch.tensor(action),
            reward=reward,
            next_state=torch.FloatTensor(next_state),
            done=done,
            log_prob=torch.tensor(log_prob)
        )
        
        # Store in memories
        self.local_memory.store_experience(
            state=state.numpy() if torch.is_tensor(state) else state,
            action=action.item() if torch.is_tensor(action) else action,
            reward=reward,
            next_state=next_state.numpy() if torch.is_tensor(next_state) else next_state,
            done=done,
            context=None,
            params=None,
            log_prob=log_prob
        )
        self.learner_memory.add(transition, tag=f"skill_{self.skill_id}")

    def _normalize_reward(self, reward: float) -> float:
        """Apply normalization to rewards"""
        if not self.reward_normalization:
            return reward
            
        # Update running statistics
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std += delta * delta2
        
        # Z-score normalization
        normalized = (reward - self.reward_mean) / (np.sqrt(self.reward_std / self.reward_count) + 1e-8)
        
        # Clipping
        return np.clip(normalized, *self.reward_clip_range)

    def _update_reward_statistics(self, reward: float):
        """Maintain running statistics for reward normalization"""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std += delta * delta2

    def compute_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """Compute discounted returns for an episode"""
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def update_policy(self):
        """Update actor-critic policy using current experiences"""
        if len(self.local_memory.episodic) == 0:
            return
            
        # Retrieve current episode experiences
        experiences = list(self.local_memory.episodic)
        
        # Extract components from dictionary experiences
        states = [e['state'] for e in experiences]
        actions = [e['action'] for e in experiences]
        rewards = [e['reward'] for e in experiences]
        dones = [e.get('done', False) for e in experiences]  # Handle missing 'done'
        
        # Extract log_probs if available
        log_probs = [e.get('log_prob', 0.0) for e in experiences]
        
        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones)
        returns_tensor = torch.FloatTensor(returns)
        
        # Compute values for advantage estimation
        state_values = []
        for state in states:
            with torch.no_grad():
                processed_state = self._process_state(state)
                state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
                value = self.actor_critic.forward_critic(state_tensor)
                state_values.append(value.item())
                
        advantages = returns_tensor - torch.FloatTensor(state_values)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.tensor(actions)
        old_log_probs_tensor = torch.tensor(log_probs)
        
        # Actor update
        self.actor_optimizer.zero_grad()
        current_log_probs = []
        entropies = []
        
        for state, action in zip(states_tensor, actions_tensor):
            processed_state = self._process_state(state.numpy())
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
            actor_output = self.actor_critic.forward_actor(state_tensor)
            dist = torch.distributions.Categorical(logits=actor_output)
            current_log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            
        current_log_probs_tensor = torch.stack(current_log_probs)
        entropy_tensor = torch.stack(entropies).mean()
        
        # Policy gradient loss
        ratio = torch.exp(current_log_probs_tensor - old_log_probs_tensor.detach())
        policy_loss = -(ratio * advantages.detach()).mean()
        
        # Entropy bonus
        entropy_loss = -self.entropy_coef * entropy_tensor
        
        # Total actor loss
        actor_loss = policy_loss + entropy_loss
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Critic update
        self.critic_optimizer.zero_grad()
        values = []
        for state in states_tensor:
            processed_state = self._process_state(state.numpy())
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0)
            values.append(self.actor_critic.forward_critic(state_tensor))
            
        values_tensor = torch.cat(values)
        critic_loss = F.mse_loss(values_tensor.squeeze(), returns_tensor)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        if self.imitation_learning and self.learner_memory.size() > 100:
            self.imitation_learning.mixed_objective_update(
                states=states_tensor,
                actions=actions_tensor,
                advantages=advantages,
                rl_loss=actor_loss + critic_loss
            )

        # Clear local memory
        self.local_memory.clear_episodic()
        
        logger.info(f"Skill {self.skill_id} updated | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")
        return actor_loss.item() + critic_loss.item()

    def save_checkpoint(self, path: str):
        """Save worker state"""
        torch.save({
            'actor_state': self.actor_critic.actor.state_dict(),
            'critic_state': self.actor_critic.critic.state_dict(),
            'actor_optim_state': self.actor_optimizer.state_dict(),
            'critic_optim_state': self.critic_optimizer.state_dict(),
            'reward_stats': (self.reward_mean, self.reward_std, self.reward_count)
        }, path)
        logger.info(f"Skill {self.skill_id} checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load worker state"""
        checkpoint = torch.load(path)
        self.actor_critic.actor.load_state_dict(checkpoint['actor_state'])
        self.actor_critic.critic.load_state_dict(checkpoint['critic_state'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state'])
        self.reward_mean, self.reward_std, self.reward_count = checkpoint['reward_stats']
        logger.info(f"Skill {self.skill_id} checkpoint loaded from {path}")

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the skill"""
        transitions = self.learner_memory.get()
        if not transitions:
            return {}
            
        rewards = [t.reward for t in transitions]
        successes = [1 if t.reward > 0 else 0 for t in transitions]  # Simple success heuristic
        
        return {
            'skill_id': self.skill_id,
            'name': self.name,
            'episode_count': len(transitions),
            'avg_reward': sum(rewards) / len(rewards),
            'success_rate': sum(successes) / len(successes),
            'recent_reward': sum(rewards[-10:]) / min(10, len(rewards))
        }

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Skill Worker ===\n")
    printer.status("TEST", "Starting Skill Worker tests", "info")
    
    # Sample skill configuration
    skill_metadata = {
        'name': 'navigate_to_A',
        'state_dim': 8,
        'action_dim': 4
    }
    
    # Create worker using factory method
    worker = SkillWorker.create_worker(skill_id=0, skill_metadata=skill_metadata)
    
    # Test action selection
    state = np.random.randn(skill_metadata['state_dim'])
    action, log_prob, entropy = worker.select_action(state)
    printer.pretty("Selected action", action, "success")
    
    # Simulate environment step
    next_state = np.random.randn(skill_metadata['state_dim'])
    reward = 1.0
    done = False
    
    # Store experience
    worker.store_experience(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        log_prob=log_prob,
        entropy=entropy
    )
    
    # Test policy update
    loss = worker.update_policy()
    printer.pretty("Update loss", loss, "success")
    
    # Test performance metrics
    metrics = worker.get_performance_metrics()
    printer.pretty("Performance metrics", metrics, "success")
    
    print("\nAll tests completed successfully!\n")
