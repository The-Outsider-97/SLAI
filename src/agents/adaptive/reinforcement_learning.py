
import numpy as np
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from typing import Any
from dataclasses import dataclass

from src.agents.learning.learning_memory import LearningMemory
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.agents.adaptive.utils.neural_network import NeuralNetwork, NeuralLayer
from logs.logger import get_logger

logger = get_logger("Reinforcement Learning")

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

@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: torch.Tensor = None  # Optional: used for policy gradients

class GoalConditionedPolicy:
    """Hierarchical policy manager for goal-oriented learning"""
    def __init__(self, state_dim, goal_dim, config):
        self.goal_network = NeuralNetwork(
            num_inputs=state_dim + goal_dim,
            layer_config=config.get('goal_conditioning', {}).get('goal_layers', []),
            problem_type='regression'
        )
        self.current_goal = np.zeros(goal_dim)
        self.goal_buffer = deque(maxlen=config.get('goal_capacity', 1000))

    def update_goal(self, state, final_reward):
        """Update high-level goal based on current state and performance"""
        goal_update = self.goal_network.feed_forward(
            np.concatenate([state, self.current_goal])
        )
        self.current_goal = 0.9 * self.current_goal + 0.1 * goal_update
        return self.current_goal

class BayesianDQN(NeuralNetwork):
    """Q-network with uncertainty estimation using MC Dropout"""
    def __init__(self, *args, dropout_rate=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        self.toggle_dropout(True)


    def toggle_dropout(self, enable=True):
        for layer in self.layers:
            layer.dropout_rate = self.dropout_rate if enable else 0.0

    def estimate_uncertainty(self, state, num_samples=10):
        """MC Dropout uncertainty estimation"""
        self.toggle_dropout(True)
        q_values = [super().feed_forward(state) for _ in range(num_samples)]
        self.toggle_dropout(False)
        return np.std(q_values, axis=0)
    
    def state_dict(self):
        return []

class ActorCriticNetwork(NeuralNetwork):
    """Policy Gradient Network with separate actor/critic outputs"""
    def __init__(self, num_inputs, actor_config, critic_config, **kwargs):
        super().__init__(num_inputs, [], **kwargs)  # Disable base network
        self.actor_layers = self._build_subnetwork(actor_config)
        self.critic_layers = self._build_subnetwork(critic_config)

    def _build_subnetwork(self, layer_config):
        layers = []
        prev_neurons = self.num_inputs
        for l_conf in layer_config:
            layer = NeuralLayer(
                num_neurons=l_conf['neurons'],
                num_inputs_per_neuron=prev_neurons,
                activation_name=l_conf['activation'],
                initialization_method=l_conf.get('init', 'he_normal')
            )
            layers.append(layer)
            prev_neurons = l_conf['neurons']
        return layers

    def forward_actor(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        x = inputs
        for layer in self.actor_layers:
            x = layer.feed_forward(x)
        return torch.tensor(x, dtype=torch.float32) if isinstance(x, list) else x

    def forward_critic(self, inputs):
        """Forward pass for value head"""
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        x = inputs
        for layer in self.critic_layers:
            x = layer.feed_forward(x)
        return x  # Returns tensor

class ReinforcementLearning(torch.nn.Module):
    """
    Reinforcement learning for self-improvement through experience
        - Sutton & Barto (2018) - Reinforcement Learning: An Introduction
        - Mnih et al. (2015) - Human-level control through deep reinforcement learning
        - Schmidhuber (2015) - On Learning to Think
    """

    def __init__(self, config, learning_memory: LearningMemory, multimodal_memory: MultiModalMemory):
        super().__init__()
        base_config = load_config() or {}
        self.config = config.get('adaptive_memory', {})
        local_memory = MultiModalMemory(config)
        self.local_memory = local_memory
        learner_memory = LearningMemory(config)
        self.learner_memory = learner_memory
        
        # Set default values for critical parameters
        self.config.setdefault('retrieval_limit', 5)
        self.config.setdefault('drift_threshold', 0.4)
        self.buffer = deque(maxlen=config.get("replay_capacity", 100000))

        # Safely merge configurations
        def safe_merge(base, user, section):
            base_section = base.get(section, {}) or {}
            user_section = user.get(section, {}) or {}
            return {**base_section, **user_section}

        self.merged_config = {
            'rl': safe_merge(base_config, config, 'rl'),
            'policy_manager': safe_merge(base_config, config, 'policy_manager'),
            'parameter_tuner': safe_merge(base_config, config, 'parameter_tuner'),
            'adaptive_memory': safe_merge(base_config, config, 'adaptive_memory')
        }

        # Neural Network setup
        state_dim = self.merged_config['rl']['state_dim']
        num_actions = self.merged_config['rl']['num_actions']
        self.num_actions = num_actions

        # Goal Conditioning
        if self.merged_config['adaptive_memory'].get('enable_goals', False):
            self.goal_policy = GoalConditionedPolicy(
                state_dim=state_dim,
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

        self.target_net = NeuralNetwork(layer_config=layer_config,
                num_inputs=state_dim,
                 loss_function_name = 'mse', # New: 'mse' or 'cross_entropy'
                 optimizer_name = 'sgd_momentum_adagrad', # New: 'sgd_momentum_adagrad', 'adam'
                 initialization_method_default = 'he_normal',
                 problem_type = 'regression', # New: 'regression' or 'classification' (binary/multiclass)
                 config = None)
        self.policy_net = BayesianDQN(
            num_inputs=state_dim,
            layer_config=layer_config,
            dropout_rate=self.merged_config['adaptive_memory'].get('uncertainty_dropout', 0.2),
            problem_type='regression'
        )

        # Policy Gradient Components
        if self.merged_config['adaptive_memory'].get('enable_policy_grad', False):
            self.actor_critic = ActorCriticNetwork(
                num_inputs=state_dim,
                actor_config=[
                    {'neurons': 64, 'activation': 'relu'},
                    {'neurons': num_actions, 'activation': 'linear'}
                ],
                critic_config=[
                    {'neurons': 64, 'activation': 'relu'},
                    {'neurons': 1, 'activation': 'linear'}
                ]
            )
            # self.value_optimizer = torch.optim.Adam(self.actor_critic.parameters())

        # Learning parameters with proper config access
        self.gamma = self.merged_config['parameter_tuner']['base_discount_factor']
        self.tau = 0.005
        self.steps_done = 0

        # Exploration parameters from config
        self.epsilon = self.merged_config['parameter_tuner']['base_exploration_rate']
        self.min_epsilon = self.merged_config['parameter_tuner']['min_exploration']
        self.epsilon_decay = self.merged_config['parameter_tuner']['exploration_decay']

        # Reward normalization parameters
        self.reward_normalization = config.get('reward_normalization', True)
        self.reward_clip_range = config.get('reward_clip_range', (-1.0, 1.0))
        self.reward_scale = config.get('reward_scale', 1.0)
        self.reward_bias = config.get('reward_bias', 0.0)
        self.reward_momentum = config.get('reward_momentum', 0.99)  # For running averages
        
        # Initialize reward statistics
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_max = 0.0
        self.reward_min = 0.0
        self.reward_count = 1e-4  # Avoid division by zero

        # Reward shaping parameters
        self.reward_shaping = config.get('reward_shaping', True)
        self.potential_scale = config.get('potential_scale', 0.1)
        self.potential_discount = config.get('potential_discount', 0.95)

        logger.info(f"Reinforcement Learning Succesfully initialized with:\n- {self.policy_net}\n- {num_actions}")

    def __len__(self):
        return len(self.buffer)

    def select_action(self, state, explore=True):
        """Enhanced action selection with uncertainty awareness and policy gradient support"""
        processed_state = state.clone().detach()
        action_info = None

        try:
            # Goal conditioning preprocessing
            if self.goal_policy:
                # Convert state tensor to numpy and flatten
                state_np = processed_state.cpu().numpy().flatten()
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
                q_values = self.policy_net.feed_forward(processed_state)
                action = np.argmax(q_values)
                action_tensor = torch.tensor([[action]], dtype=torch.long)

            # Policy gradient action sampling
            if self.merged_config['adaptive_memory'].get('enable_policy_grad', False):
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

    def _normalize_reward(self, reward):
        """Apply normalization and scaling to rewards"""
        if not self.reward_normalization:
            return reward * self.reward_scale + self.reward_bias
            
        # Z-score normalization
        normalized = (reward - self.reward_mean) / (np.sqrt(self.reward_std/self.reward_count) + 1e-8)
        
        # Clipping
        normalized = np.clip(normalized, *self.reward_clip_range)
        
        return normalized * self.reward_scale + self.reward_bias

    def _calculate_potential(self, state):
        """Multi-modal potential calculation with configurable strategies"""
        potential_type = self.config.get('potential_type', 'l2_norm')
        
        if not self.reward_shaping:
            return 0.0
    
        # Convert tensor to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        state = state.flatten()  # Ensure 1D array
        
        if potential_type == 'l2_norm':
            return self._l2_potential(state)
        elif potential_type == 'goal_based':
            return self._goal_potential(state)
        elif potential_type == 'feature_based':
            return self._feature_potential(state)
        elif potential_type == 'learned':
            return self._learned_potential(state)
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")
    
    def _l2_potential(self, state):
        """Basic distance potential"""
        state_norm = np.linalg.norm(state)
        return self.potential_scale * state_norm / (1.0 + state_norm)
    
    def _goal_based_potential(self, state):
        """Goal-oriented potential (requires goal conditioning)"""
        if not self.goal_policy:
            return 0.0
            
        goal_diff = state[-self.goal_policy.current_goal.size:] - self.goal_policy.current_goal
        return -self.potential_scale * np.linalg.norm(goal_diff)
    
    def _feature_based_potential(self, state):
        """Key feature potential (configure features in config)"""
        features = self.config.get('potential_features', [0])
        selected = state[features]
        return self.potential_scale * np.mean(selected)

    def _goal_potential(self, state):
        return []

    def _feature_potential(self, state):
        return []

    def _learned_potential(self, state):
        """Neural network-based potential"""
        return self.potential_scale * self.goal_policy.goal_network.feed_forward(state)

    def _apply_reward_shaping(self, state, next_state, reward):
        """Apply potential-based reward shaping"""
        if not self.reward_shaping:
            return reward
            
        current_potential = self._calculate_potential(state)
        next_potential = self._calculate_potential(next_state)
        return reward + self.potential_discount * next_potential - current_potential

    def update_policy(self):
        """Train policy network using experiences from memory."""
        if self.learner_memory.size() < self.merged_config['rl'].get('batch_size', 64):
            return None
    
        transitions = self.learner_memory.sample(self.merged_config['rl'].get('batch_size', 64))
        batch = Transition(*zip(*transitions))
    
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
            next_q = max(self.target_net.feed_forward(next_state_batch[i]))
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
    print("\n=== Running Reinforcement Learning ===\n")

    config = {
        'rl': {
            'state_dim': 10,
            'num_actions': 2,
            'batch_size': 64
        },
        'adaptive_memory': {
            'enable_goals': True,
            'goal_dim': 16,
            'uncertainty_dropout': 0.2,
            # ... other memory params
        },
        'policy_manager': {
            'hidden_layer_sizes': [64, 32],  # Matches PolicyNetwork's expected key
            'activation': 'tanh',
            'output_activation': 'linear'
        },
        'parameter_tuner': {
            'base_learning_rate': 0.001,
            'base_exploration_rate': 0.3,
            'min_exploration': 0.01,
            'exploration_decay': 1000,
            'base_discount_factor': 0.95
        }
    }
    learning_memory=LearningMemory(config)
    multimodal_memory=MultiModalMemory(config)
    explore=True
    state = np.random.rand(config['rl']['state_dim']).astype(np.float32)

    agent = ReinforcementLearning(
        config=config,
        learning_memory=learning_memory,
        multimodal_memory=multimodal_memory
    )
    dummy_state = torch.randn(1, config['rl']['state_dim'])
    action_output = agent.select_action(dummy_state, explore=True)
    
    if isinstance(action_output, tuple):
        action, log_prob, entropy = action_output
        print(f"Selected action: {action.item()} | Log prob: {log_prob:.3f}")
    else:
        print(f"Selected action: {action_output.item()}")
    loss = agent.update_policy()


    print(f"Selected action: {action.item()}")
    print(f"Policy update loss: {loss if loss else 'No update'}")

    print(f"\n* * * * * Phase 2 * * * * *\n")
    
    update = agent._update_target_network()

    print(f"{update}")

    print(f"\n* * * * * Phase 3 * * * * *\n")
    import gym
    from collections import namedtuple

    env = gym.make("CartPole-v1")
    config = load_config()
    
    # --- RL agent and memory ---
    learner_memory = LearningMemory(config)
    adaptive_memory = MultiModalMemory(config)
    agent = ReinforcementLearning(config, learner_memory, adaptive_memory)
    
    # --- Training parameters ---
    num_episodes = 100
    max_steps_per_episode = 250
    
    # --- Training loop ---
    for episode in range(num_episodes):
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})  # Support Gym >= 0.26
        state = torch.tensor(obs, dtype=torch.float32)
        total_reward = 0
    
        for step in range(max_steps_per_episode):
            action_output = agent.select_action(state, explore=True)
            action_tensor = action_output[0] if isinstance(action_output, tuple) else action_output
            action = int(action_tensor.item())
    
            # Step in environment
            result = env.step(action)
            if len(result) == 5:
                next_obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = result
    
            next_state = torch.tensor(next_obs, dtype=torch.float32)
    
            # Save transition
            transition = Transition(
                state=state,
                action=torch.tensor([action], dtype=torch.long),
                reward=reward,
                next_state=next_state,
                done=done
            )
            learner_memory.add(transition)
    
            # Update policy
            agent.update_policy()
    
            # Prepare for next step
            state = next_state
            total_reward += reward
    
            if done:
                break
    
        print(f"Episode {episode + 1} - Total Reward: {total_reward}")
    
    env.close()
    print("\n=== Successfully Ran Reinforcement Learning ===\n")
