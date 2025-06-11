
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
from typing import Any

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.learning_memory import LearningMemory
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.agents.adaptive.utils.neural_network import NeuralNetwork, BayesianDQN, ActorCriticNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reinforcement Learning")
printer = PrettyPrinter

@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: torch.Tensor = None  # Optional: used for policy gradients

class GoalConditionedPolicy(nn.Module):
    """Hierarchical policy manager for goal-oriented learning"""
    def __init__(self, state_dim, goal_dim, config):
        super().__init__()
        # Extract goal network configuration
        goal_conf = config.get('goal_conditioning', {})
        goal_layers = goal_conf.get('goal_layers', [])
        hidden_sizes = [layer['neurons'] for layer in goal_layers[:-1]] if goal_layers else []
        output_size = goal_layers[-1]['neurons'] if goal_layers else goal_dim

        self.current_goal = np.zeros(goal_dim)
        self.goal_buffer = deque(maxlen=config.get('goal_capacity', 1000))
        self.state_dim = state_dim
        self.goal_dim = goal_dim

    def goal_network(self, input_dim, output_dim, hidden_layers):
        layers = []
        current_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            current_dim = h
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def update_goal(self, state, final_reward):
        """Update high-level goal based on current state and performance"""
        # Ensure state is 1D array
        state = state.flatten() if isinstance(state, np.ndarray) else state.cpu().numpy().flatten()
        
        # Create input tensor
        input_arr = np.concatenate([state[:self.state_dim], self.current_goal])
        input_tensor = torch.tensor(input_arr, dtype=torch.float32).unsqueeze(0)
        
        # Get goal update
        with torch.no_grad():
            goal_update = self.goal_network(input_tensor).squeeze(0).numpy()
        
        self.current_goal = 0.9 * self.current_goal + 0.1 * goal_update
        return self.current_goal

class ReinforcementLearning(torch.nn.Module):
    """
    Reinforcement learning for self-improvement through experience
        - Sutton & Barto (2018) - Reinforcement Learning: An Introduction
        - Mnih et al. (2015) - Human-level control through deep reinforcement learning
        - Schmidhuber (2015) - On Learning to Think
    """

    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.rl_config = get_config_section('adaptive_memory')
        self.retrieval_limit = self.rl_config.get('retrieval_limit', 5)
        self.drift_threshold = self.rl_config.get('drift_threshold', 0.4)

        self.local_memory = MultiModalMemory()  # For policy evolution & prioritized replay
        self.learner_memory = LearningMemory()  # For checkpointing & long-term storage

        # Create merged configuration by combining sections
        self.merged_config = {
            'rl': self._get_config_section('rl'),
            'policy_manager': self._get_config_section('policy_manager'),
            'parameter_tuner': self._get_config_section('parameter_tuner'),
            'adaptive_memory': self._get_config_section('adaptive_memory')
        }

        # Neural Network setup
        state_dim = self.merged_config['rl'].get('state_dim', 4)
        num_actions = self.merged_config['rl'].get('num_actions', 2)
        self.num_actions = num_actions

        # Goal Conditioning
        if self.merged_config['adaptive_memory'].get('enable_goals', False):
            self.goal_policy = GoalConditionedPolicy(
                state_dim=self.merged_config['rl'].get('state_dim', 4),  # Add state_dim
                goal_dim=self.merged_config['adaptive_memory'].get('goal_dim', 16),
                config=self.merged_config['adaptive_memory']
            )
            state_dim += self.merged_config['adaptive_memory'].get('goal_dim', 16)
        else:
            self.goal_policy = None

        # BayesianDQN with uncertainty
        layer_config = [
            {'neurons': 128, 'activation': 'relu', 'init': 'he_normal',
             'dropout': self.merged_config['adaptive_memory'].get('uncertainty_dropout', 0.2)},
            {'neurons': 64, 'activation': 'relu', 'init': 'he_normal'},
            {'neurons': num_actions, 'activation': 'linear'}
        ]

        # BayesianDQN with uncertainty
        self.target_net = NeuralNetwork()
        self.policy_net = BayesianDQN(
            dropout_rate=self.merged_config['adaptive_memory'].get('uncertainty_dropout', 0.2)
            # problem_type='regression'
        )

        # Policy Gradient Components
        if self.merged_config['adaptive_memory'].get('enable_policy_grad', False):
            self.actor_critic = ActorCriticNetwork(
                state_dim=state_dim,  # Updated parameter name
                action_dim=num_actions,  # Updated parameter name
                actor_layers=[64, num_actions],  # Use layer sizes instead of config
                critic_layers=[64, 1]  # Use layer sizes instead of config
            )

        # Learning parameters with proper config access
        self.gamma = self.merged_config['parameter_tuner'].get('base_discount_factor', 0.95)
        self.tau = 0.005
        self.steps_done = 0

        # Exploration parameters from config
        self.epsilon = self.merged_config['parameter_tuner'].get('base_exploration_rate', 0.3)
        self.min_epsilon = self.merged_config['parameter_tuner'].get('min_exploration', 0.01)
        self.epsilon_decay = self.merged_config['parameter_tuner'].get('exploration_decay', 0.9995)

        # Reward normalization parameters
        self.reward_normalization = self.config.get('reward_normalization', True)
        self.reward_clip_range = self.config.get('reward_clip_range', (-1.0, 1.0))
        self.reward_scale = self.config.get('reward_scale', 1.0)
        self.reward_bias = self.config.get('reward_bias', 0.0)
        self.reward_momentum = self.config.get('reward_momentum', 0.99)
        
        # Initialize reward statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_max = 0.0
        self.reward_min = 0.0
        self.reward_count = 1e-4

        # Reward shaping parameters
        self.reward_shaping = self.config.get('reward_shaping', True)
        self.potential_scale = self.config.get('potential_scale', 0.1)
        self.potential_discount = self.config.get('potential_discount', 0.95)

        logger.info(f"Reinforcement Learning Succesfully initialized with:\n- {self.policy_net}\n- {num_actions}")

    def __len__(self):
        return len(self.buffer)

    def _get_config_section(self, section_name: str) -> dict:
        """Helper to get configuration section with fallback to empty dict"""
        section = get_config_section(section_name)
        return section if section is not None else {}

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in both memory systems"""
        # Store in multimodal memory with self-tuning prioritization
        self.local_memory.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        # Also store in learning memory for long-term persistence
        self.learner_memory.add(
            (state, action, reward, next_state, done),
            tag="rl_experience"
        )

    def select_action(self, state, explore=True):
        """Enhanced action selection with uncertainty awareness and policy gradient support"""
        processed_state = state.clone().detach()
        action_info = None

        try:
            # Goal conditioning preprocessing
            if self.goal_policy:
                state_np = state.cpu().numpy().flatten()
                processed_state = np.concatenate([state_np, self.goal_policy.current_goal])
                if not isinstance(processed_state, np.ndarray):
                    processed_state = processed_state.numpy()

            # Exploration vs exploitation logic
            if explore and random.random() < self.epsilon:
                # Uncertainty-driven exploration
                uncertainty = self.policy_net.estimate_uncertainty(processed_state)

                # Ensure weights are non-negative and have a positive sum
                uncertainty = np.clip(uncertainty, a_min=1e-6, a_max=None)
                if np.sum(uncertainty) == 0 or np.any(np.isnan(uncertainty)):
                    logger.warning("Invalid uncertainty weights detected; falling back to uniform random choice.")
                    action = random.randint(0, self.num_actions - 1)
                else:
                    action = random.choices(range(self.num_actions), weights=uncertainty)[0]
                action_tensor = torch.tensor([[action]], dtype=torch.long)
            else:
                # Q-value based exploitation
                with torch.no_grad():
                    state_tensor = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)
                    q_values = self.policy_net(state_tensor).squeeze(0).numpy()
                action = np.argmax(q_values)
                action_tensor = torch.tensor([[action]], dtype=torch.long)

            # Policy gradient action sampling
            if self.merged_config['adaptive_memory'].get('enable_policy_grad', False):
                with torch.no_grad():
                    state_tensor = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)
                    actor_output = self.actor_critic.forward_actor(state_tensor).squeeze(0)
                actor_output = self.actor_critic.forward_actor(processed_state)
                if isinstance(actor_output, list):
                    actor_output = torch.tensor(actor_output, dtype=torch.float32)
                probs = F.softmax(actor_output, dim=-1)
                dist = torch.distributions.Categorical(probs)
                sampled_action = dist.sample()
                return sampled_action, dist.log_prob(sampled_action), dist.entropy()

            return action_tensor

        except Exception as e:
            logger.error(f"Action selection failed: {str(e)}")
            # Fallback to random action with error logging
            fallback_action = torch.tensor([[random.randint(0, self.num_actions-1)]], 
                                        dtype=torch.long)
            if self.merged_config['adaptive_memory'].get('enable_policy_grad', False):
                return fallback_action, torch.tensor(0.0), torch.tensor(0.0)
            return fallback_action

    def _normalize_reward(self, reward):
        """Apply normalization and scaling to rewards"""
        if not self.reward_normalization:
            return reward * self.reward_scale + self.reward_bias
            
        # Z-score normalization
        normalized = (reward - self.reward_mean) / (np.sqrt(self.reward_std/self.reward_count) + 1e-8)
        
        # Clipping
        normalized = np.clip(normalized, *self.reward_clip_range)
        
        return normalized * self.reward_scale + self.reward_bias

    def _apply_reward_shaping(self, state, next_state, reward):
        """Apply potential-based reward shaping"""
        if not self.reward_shaping:
            return reward
            
        current_potential = self._calculate_potential(state)
        next_potential = self._calculate_potential(next_state)
        return reward + self.potential_discount * next_potential - current_potential

    def _calculate_potential(self, state):
        """Calculate potential-based reward shaping value using configured strategy
        
        Args:
            state: Current environment state (tensor or numpy array)
            
        Returns:
            Potential value (float)
            
        Raises:
            ValueError: For unknown potential types
            RuntimeError: For goal-based potential without goal policy
        """
        potential_type = self.config.get('potential_type', 'l2_norm')
        
        # Bypass calculation if reward shaping is disabled
        if not self.reward_shaping:
            return 0.0
    
        # Convert and validate input state
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        if not isinstance(state, np.ndarray):
            logger.warning(f"Invalid state type for potential calculation: {type(state)}")
            return 0.0
            
        # Ensure 1D array for consistent processing
        original_shape = state.shape
        state = state.flatten()
        if len(state) == 0:
            logger.error("Received empty state for potential calculation")
            return 0.0
            
        try:
            # Dispatch to specific potential calculation method
            if potential_type == 'l2_norm':
                return self._l2_potential(state)
            elif potential_type == 'goal_based':
                return self._goal_based_potential(state)
            elif potential_type == 'feature_based':
                return self._feature_based_potential(state)
            elif potential_type == 'learned':
                return self._learned_potential(state)
            else:
                raise ValueError(f"Unknown potential type: {potential_type}")
                
        except Exception as e:
            logger.error(f"Potential calculation failed: {str(e)}")
            # Return safe default on error
            return 0.0
    
    def _l2_potential(self, state: np.ndarray) -> float:
        """Calculate L2 norm-based potential
        
        Formula: potential_scale * ||state|| / (1 + ||state||)
        Provides diminishing returns as state norm increases
        """
        try:
            state_norm = np.linalg.norm(state)
            # Add epsilon to avoid division by zero
            return self.potential_scale * state_norm / (1.0 + state_norm + 1e-8)
        except Exception as e:
            logger.error(f"L2 potential calculation failed: {str(e)}")
            return 0.0
    
    def _goal_based_potential(self, state: np.ndarray) -> float:
        """Calculate goal-oriented potential
        
        Requires:
            - Goal policy is enabled
            - State contains goal information at the end
            
        Formula: -potential_scale * ||state[goal_slice] - current_goal||
        """
        if not self.goal_policy:
            logger.warning("Goal-based potential requested but goal policy is disabled")
            return 0.0
            
        try:
            # Extract goal portion from state
            goal_size = self.goal_policy.current_goal.size
            if state.size < goal_size:
                logger.error(f"State size {state.size} smaller than goal size {goal_size}")
                return 0.0
                
            state_goal = state[-goal_size:]
            goal_diff = state_goal - self.goal_policy.current_goal
            
            # Calculate distance to current goal
            goal_distance = np.linalg.norm(goal_diff)
            return -self.potential_scale * goal_distance
        except Exception as e:
            logger.error(f"Goal-based potential calculation failed: {str(e)}")
            return 0.0
    
    def _feature_based_potential(self, state: np.ndarray) -> float:
        """Calculate potential based on key features
        
        Requires:
            - 'potential_features' list in config
            - Features must be valid indices for the state array
        """
        features = self.config.get('potential_features', [])
        if not features:
            logger.warning("Feature-based potential requested but no features configured")
            return 0.0
            
        try:
            # Validate feature indices
            valid_features = [idx for idx in features if idx < len(state)]
            if not valid_features:
                logger.error(f"No valid features in {features} for state size {len(state)}")
                return 0.0
                
            # Calculate mean of selected features
            feature_values = state[valid_features]
            return self.potential_scale * np.mean(feature_values)
        except Exception as e:
            logger.error(f"Feature-based potential calculation failed: {str(e)}")
            return 0.0
    
    def _learned_potential(self, state: np.ndarray) -> float:
        """Calculate potential using neural network prediction
        
        Requires:
            - Goal policy with initialized network
        """
        if not self.goal_policy:
            logger.warning("Learned potential requested but goal policy is disabled")
            return 0.0
            
        try:
            # Convert to tensor and add batch dimension
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                potential = self.goal_policy.goal_network(state_tensor).item()
                
            return self.potential_scale * potential
        except Exception as e:
            logger.error(f"Learned potential calculation failed: {str(e)}")
            return 0.0

    def _process_policy_batch(self):
        """Process batch for policy gradient training"""
        batch_size = self.merged_config['rl'].get('batch_size', 64)
        batch = self.learner_memory.sample(batch_size)
        returns = self._calculate_returns([t.reward for t in batch])
        return (
            [t.state for t in batch],
            [t.action for t in batch], 
            [t.log_prob for t in batch],
            returns
        )

    def _calculate_returns(self, rewards):
        """Calculate discounted returns for policy gradient"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def _update_target_network(self):
        """Soft/hard update using the NeuralNetwork's custom parameter methods."""
        # Configurable parameters
        hard_update_interval = self.merged_config['parameter_tuner'].get('target_update_interval', 100)
        update_type = 'hard' if self.steps_done % hard_update_interval == 0 else 'soft'
    
        if update_type == 'hard':
            # Hard update: full parameter replacement
            policy_params = self.policy_net.get_weights_biases()
            self.target_net.set_weights_biases(policy_params)
            logger.debug(f"Hard target network update at step {self.steps_done}")
        else:
            # Soft update: interpolate parameters with tau
            policy_params = self.policy_net.get_weights_biases()
            target_params = self.target_net.get_weights_biases()
            
            # Iterate through each layer and neuron to apply tau mixing
            for layer_idx, (policy_layer, target_layer) in enumerate(zip(policy_params, target_params)):
                for neuron_idx, (policy_neuron, target_neuron) in enumerate(zip(policy_layer['neurons'], target_layer['neurons'])):
                    # Update weights
                    blended_weights = [
                        self.tau * p_w + (1 - self.tau) * t_w
                        for p_w, t_w in zip(policy_neuron['weights'], target_neuron['weights'])
                    ]
                    # Update bias
                    blended_bias = self.tau * policy_neuron['bias'] + (1 - self.tau) * target_neuron['bias']
                    
                    # Apply to target network
                    self.target_net.layers[layer_idx].neurons[neuron_idx].weights = blended_weights
                    self.target_net.layers[layer_idx].neurons[neuron_idx].bias = blended_bias
    
            logger.debug(f"Soft target network update at step {self.steps_done}")
    
        # Track update statistics
        self._log_update_metrics(update_type)
        self.steps_done += 1
    
    def _log_update_metrics(self, update_type: str):
        import logging
        """Log detailed update statistics."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
            
        sample_weight = next(iter(self.target_net.parameters()))[0] if isinstance(self.target_net, nn.Module) else 0
        stats = {
            'update_type': update_type,
            'steps': self.steps_done,
            'tau': self.tau,
            'sample_weight': sample_weight,
            'target_network': str(self.target_net)[:100]  # Truncate network string
        }
        logger.debug("Target Network Update Stats", extra=stats)

    def update_policy(self):
        """Train policy network using experiences from memory."""
        if self.learner_memory.size() < self.merged_config['rl'].get('batch_size', 64):
            return None

        transitions = self.learner_memory.sample(self.merged_config['rl'].get('batch_size', 64))
        batch = Transition(*zip(*transitions))

        batch_size = self.merged_config['rl'].get('batch_size', 64)
        if self.local_memory.replay_buffer.size() < batch_size:
            return None

        samples = self.local_memory.replay_buffer.sample(agent_id="default", batch_size=batch_size)

        processed_batch = []
        for transition in transitions:
            # Convert to numpy arrays
            state = transition.state.cpu().numpy()
            next_state = transition.next_state.cpu().numpy()
            reward = transition.reward

            # Update reward statistics
            self._update_reward_statistics(reward)

            # Apply reward shaping
            shaped_reward = self._apply_reward_shaping(state, next_state, reward)

            # Apply normalization
            normalized_reward = self._normalize_reward(shaped_reward)

            # Create new transition with processed reward
            processed_transition = Transition(
                state=transition.state,
                action=transition.action,
                reward=normalized_reward,
                next_state=transition.next_state,
                done=transition.done
            )
            processed_batch.append(processed_transition)

        # Convert batch to lists
        state_batch = [s.tolist() for s in batch.state]
        next_state_batch = [s.tolist() for s in batch.next_state]
        target_q_values = []
        for i in range(len(transitions)):
            with torch.no_grad():
                next_state_tensor = torch.tensor(next_state_batch[i], dtype=torch.float32).unsqueeze(0)
                next_q = self.target_net(next_state_tensor).max().item()
            target_q = batch.reward[i] + self.gamma * next_q
            target_q_values.append([target_q])

        # Train policy network on the computed targets
        training_data = list(zip(state_batch, target_q_values))
        self.policy_net.train(
            training_data,
            epochs=1,
            initial_learning_rate=self.merged_config['parameter_tuner']['base_learning_rate'],  # Use merged_config
            batch_size=self.merged_config['rl'].get('batch_size', 64),  # Use merged_config
            verbose=False
        )

        # Update target network
        self._update_target_network()

        # Policy Gradient Update
        if self.config['enable_policy_grad']:
            states, actions, old_log_probs, returns = self._process_policy_batch()
            
            # Calculate advantages
            values = torch.tensor([self.actor_critic.forward_critic(s)[0] for s in states])
            advantages = returns - values
            
            # Update actor
            probs = F.softmax(torch.tensor([self.actor_critic.forward_actor(s) for s in states]), dim=-1)
            dists = torch.distributions.Categorical(probs)
            policy_loss = -torch.mean(dists.log_prob(actions) * advantages)
            
            # Update critic
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.value_optimizer.step()

        total_loss = 0.0  # Replace combined_loss with actual loss tracking
        total_loss += self.policy_net.loss
        if self.merged_config['adaptive_memory'].get('enable_policy_grad', False):
            total_loss += (policy_loss + value_loss).item()

        # Goal-conditioned learning update
        if self.goal_policy:
            for transition in processed_batch:
                self.goal_policy.update_goal(transition.state, transition.reward)

        return total_loss 

    def _update_reward_statistics(self, reward):
        """Maintain running statistics for reward normalization"""
        # Update count first
        self.reward_count += 1
        
        # Update mean and std using Welford's algorithm
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std += delta * delta2

        # Update min/max with momentum
        self.reward_max = self.reward_momentum * self.reward_max + (1 - self.reward_momentum) * max(reward, self.reward_max)
        self.reward_min = self.reward_momentum * self.reward_min + (1 - self.reward_momentum) * min(reward, self.reward_min)

    def log_parameters(self, performance, params):
        """Track evolution of learning parameters"""
        self.local_memory.log_parameters(performance, params)

    def save_checkpoint(self, path):
        """Save complete agent state"""
        # Save neural network weights
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict()
        }, path + "_networks.pth")
        
        # Save memory states
        self.local_memory.consolidate()
        self.learner_memory.save_checkpoint(path + "_memory.pt")
        
        # Save agent configuration
        with open(path + "_config.yaml", 'w') as f:
            yaml.dump({
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma
            }, f)

    def load_checkpoint(self, path):
        """Load complete agent state"""
        # Load networks
        checkpoint = torch.load(path + "_networks.pth")
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        
        # Load memories
        self.learner_memory.load_checkpoint(path + "_memory.pt")
        
        # Load configuration
        with open(path + "_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            self.epsilon = config['epsilon']
            self.learning_rate = config['learning_rate']
            self.gamma = config['gamma']

    def get_memory_report(self):
        """Get combined memory analytics"""
        mm_report = self.local_memory.get_memory_report()
        lm_metrics = self.learner_memory.metrics()
        
        return {
            **mm_report,
            "learning_memory": lm_metrics,
            "total_experiences": lm_metrics['size'] + mm_report['replay_stats']['size']
        }

# ===== Helpers =====
def to_tensor(state):
    if isinstance(state, torch.Tensor):
        return state.float()
    elif isinstance(state, np.ndarray):
        return torch.from_numpy(state.astype(np.float32))
    elif isinstance(state, tuple):
        return torch.tensor(np.array(state), dtype=torch.float32)
    elif isinstance(state, list):
        return torch.tensor(state, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported state format: {type(state)}")

def preprocess_state(env_reset_output):
    obs = env_reset_output
    if isinstance(obs, tuple):
        obs = obs[0]
    return torch.tensor(obs, dtype=torch.float32)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Adaptive Reinforcement Learning ===\n")
    printer.status("TEST", "Starting Reinforcement Learning tests", "info")

    agent = ReinforcementLearning()
    print(f"Main: {agent}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    state= to_tensor([4.0])
    processed_state = state.clone().detach()
    explore=True
    reward=10

    action = agent.select_action(state=state, explore=explore)

    printer.pretty("select", action, "success")
    printer.pretty("based", agent._feature_based_potential(state=state), "success")
    printer.pretty("update", agent._update_reward_statistics(reward=reward), "success")
    printer.pretty("norm", agent._normalize_reward(reward=reward), "success")
    printer.pretty("potential", agent._calculate_potential(state=state), "success")
    printer.pretty("batch", agent._process_policy_batch(), "success")
    printer.pretty("policy", agent.update_policy(), "success")
    print("\n=== Successfully Ran Adaptive Reinforcement Learning ===\n")
