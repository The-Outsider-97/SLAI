__version__ = "1.8.0"

"""
Adaptive Agent with Reinforcement Learning Capabilities

This agent combines:
1. Reinforcement learning for self-improvement through experience
2. Memory systems for retaining knowledge
3. Adaptive routing for task delegation
4. Continuous learning from feedback and demonstrations

Key Features:
- Self-tuning learning parameters
- Multi-modal memory system
- Flexible task routing
- Integrated learning from various sources
- Minimal external dependencies

Academic References:
- Sutton & Barto (2018) - Reinforcement Learning: An Introduction
- Silver et al. (2014) - Deterministic Policy Gradient Algorithms
- Mnih et al. (2015) - Human-level control through deep reinforcement learning
- Schmidhuber (2015) - On Learning to Think
"""

import random
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import statsmodels.formula.api as smf

from datetime import timedelta
from collections import defaultdict, deque

from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.agents.adaptive.policy_manager import PolicyManager
from src.agents.adaptive.parameter_tuner import LearningParameterTuner
from src.agents.adaptive.reinforcement_learning import ReinforcementLearning, Transition
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger

logger = get_logger("Adaptive Agent")

LOCAL_ALIGNMENT_CONFIG_PATH = "src/agents/alignment/configs/alignment_config.yaml"
LOCAL_ADAPTIVE_CONFIG_PATH = "src/agents/adaptive/configs/adaptive_config.yaml"

class AdaptiveAgent(BaseAgent):
    """
    An adaptive agent that combines reinforcement learning with memory and routing capabilities.
    Continuously improves from feedback, success/failure, and demonstrations.
    """

    def __init__(self, shared_memory, agent_factory, config=None, args=(), kwargs={}):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        """
        Initialize the adaptive agent with learning and memory systems.
        """
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self.state_dim = self.config.get('rl', {}).get('state_dim', self.config.get('state_dim', 10))
        self.num_actions = self.config.get('rl', {}).get('num_actions', self.config.get('num_actions', 2))
        self.num_handlers = self.config.get('rl', {}).get('num_handlers', self.config.get('num_handlers', 3))


        self.episodic_memory = deque(maxlen=1000)  # Recent experiences
        self.semantic_memory = defaultdict(dict)   # Conceptual knowledge
        self.rl_engine = ReinforcementLearning()

        self.recent_rewards = deque(maxlen=100)
        self.episode_reward = 0.0
        self.training = False
        self.recovery_history = defaultdict(lambda: {'success': 0, 'fail': 0})
        self.recovery_strategies = [self._recover_soft_reset, self._recover_lr_adjustment, self._recover_full_reset]
        self.error_classes = {
            'ValueError': 'input_error',
            'KeyError': 'missing_data',
            'TypeError': 'type_mismatch',
            'RuntimeError': 'runtime_failure',
            'IndexError': 'index_problem',
            'ZeroDivisionError': 'math_error',
            # Add more as you identify patterns
        }

        self.routing_policy = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_handlers)
        )
        self.routing_optimizer = torch.optim.Adam(self.routing_policy.parameters())

        # Policy and value function representations
        self.value_estimates = defaultdict(float)
        
        # Performance tracking
        self.performance_history = []
        self.last_reward = 0
        self.total_steps = 0
        
        # Skill library
        self.skills = {
            'basic_rl': self._basic_rl_skill,
            'memory_retrieval': self._memory_skill,
            'message_routing': self._routing_skill
        }

        # Add priority experience replay parameters
        self.per_beta = config.get('per_beta', 0.4)
        self.per_epsilon = config.get('per_epsilon', 1e-6)

        self._init_subsystems(learning_params=None)

        
        logger.info("AdaptiveAgent initialized with skills: %s", self.skills)

    def _init_subsystems(self, learning_params=None):
        # Core subsystems
        from src.agents.collaborative.task_router import AdaptiveRouter
        from src.utils.buffer.distributed_replay_buffer import DistributedReplayBuffer

        self.learning_params = {
            'learning_rate': 0.01,
            'exploration_rate': 0.3,
            'discount_factor': 0.95,
            'temperature': 1.0,  # For softmax decision making
            'memory_capacity': 1000,
            **(learning_params or {}) # Use learning_params if provided, else empty dict
        }

        self.memory = MultiModalMemory()
        self.tuner = LearningParameterTuner(self.learning_params) # Pass the initialized learning_params
        self.router = AdaptiveRouter()
        self.policy = PolicyManager(self.state_dim, self.num_actions)

        self.replay_buffer = DistributedReplayBuffer(
            user_config={
                'distributed': {
                    'capacity': self.config.get('replay_capacity', 100000),
                    'prioritization_alpha': self.config.get('priority_alpha', 0.6),
                    'staleness_threshold_days': self.config.get('experience_staleness_days', 1)
                }
            }
        )

        logger.info("Subcomponents initialized with parameters: %s", self.learning_params)

    def _store_experience(self, state, action, reward):
        """Store experience in memory systems with type checking."""
        # Validate reward type
        if not isinstance(reward, (int, float)):
            raise TypeError(f"Reward must be numeric. Received type: {type(reward)}")
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.episodic_memory.append(experience)
        
        # Store in semantic memory if significant
        if abs(reward) > 1.0:
            key = f"state_{hash(state) % 1000}"
            self.semantic_memory[key][action] = reward

    def update_recovery_success(self, strategy_name, success=True):
        if success:
            self.recovery_history[strategy_name]['success'] += 1
        else:
            self.recovery_history[strategy_name]['fail'] += 1

    def rank_recovery_strategies(self):
        return sorted(
            self.recovery_strategies,
            key=lambda s: self.recovery_history[s.__name__]['success'] /
                        (self.recovery_history[s.__name__]['success'] +
                        self.recovery_history[s.__name__]['fail'] + 1),
            reverse=True
        )

    def _take_action(self, state, action):
        """
        Simulates complex environment dynamics with multiple state dimensions
        and realistic physical constraints.
        
        State Components:
        0: x_position (0-100) 
        1: y_position (0-100)
        2: velocity (0-10)
        3: energy (0-100)
        4: cargo (0-50)
        5: integrity (0-100)
        6: temperature (-50-100)
        
        Actions:
        0: Accelerate
        1: Decelerate
        2: Turn left
        3: Turn right
        4: Collect resource
        5: Repair system
        6: Cool system
        """
        # Convert state to mutable list
        current_state = list(state) # Use a different name to avoid confusion with outer scope 'state'
        reward = 0
        done = False
        action_success = True
        stochasticity = random.random()
        
        # Physical constraints
        MAX_VELOCITY = 10
        MAX_ENERGY = 100
        MAX_TEMP = 100
        CARGO_CAPACITY = 50
        
        # Action effects with stochastic outcomes
        try:
            if action == 0:  # Accelerate
                if current_state[3] > 5 and current_state[6] < 80:
                    current_state[2] = min(current_state[2] + 1 + 0.5*stochasticity, MAX_VELOCITY)
                    current_state[3] -= 3 * (1 + current_state[2]/MAX_VELOCITY)
                    current_state[6] += 2
                    reward += current_state[2] * 0.2
                else:
                    action_success = False
                    reward -= 1
    
            elif action == 1:  # Decelerate
                current_state[2] = max(current_state[2] - 1 - stochasticity, 0)
                current_state[6] -= 1
                reward += current_state[2] * 0.1
    
            elif action == 2:  # Turn left
                new_y = current_state[1] + current_state[2] * (0.8 + stochasticity*0.4) # Example: move along y-axis for left/right
                if 0 <= new_y <= 100:
                    current_state[1] = new_y
                    reward += abs(current_state[2]) * 0.05
                else:
                    reward -= 2  # Wall collision
    
            elif action == 3:  # Turn right
                new_x = current_state[0] + current_state[2] * (0.8 + stochasticity*0.4) # Example: move along x-axis
                if 0 <= new_x <= 100:
                    current_state[0] = new_x
                    reward += abs(current_state[2]) * 0.05
                else:
                    reward -= 2  # Wall collision
    
            elif action == 4:  # Collect resource
                if current_state[4] < CARGO_CAPACITY and current_state[3] > 10:
                    collection = min(3 * (1 + stochasticity), CARGO_CAPACITY - current_state[4])
                    current_state[4] += collection
                    current_state[3] -= 5
                    reward += collection * 0.7
                    current_state[6] += 1
                else:
                    action_success = False
                    reward -= 0.5
    
            elif action == 5:  # Repair system
                if current_state[3] > 20:
                    repair_amount = 10 * (0.5 + stochasticity)
                    current_state[5] = min(current_state[5] + repair_amount, 100)
                    current_state[3] -= 15
                    reward += repair_amount * 0.3
                    current_state[6] += 2
                else:
                    action_success = False
    
            elif action == 6:  # Cool system
                cooling = 5 * (0.8 + stochasticity*0.4)
                current_state[6] = max(current_state[6] - cooling, -50)
                current_state[3] -= 2
                reward += cooling * 0.2
    
            # Environmental dynamics
            current_state[3] = max(current_state[3] - 0.1 * current_state[2], 0)  # Energy drain
            current_state[6] += 0.05 * current_state[2]**2  # Friction heating
            if random.random() < 0.1:  # Random environmental events
                current_state[6] += 5 * random.random()
                
        except Exception as e:
            logger.error(f"Action execution error: {str(e)}")
            reward -= 5
            action_success = False
    
        # Terminal conditions
        done = any([
            current_state[3] <= 0,  # No energy
            current_state[5] <= 0,  # System failure
            current_state[6] >= MAX_TEMP,  # Overheating
            (current_state[0] >= 95 and current_state[1] >= 95)  # Reached target zone
        ])
        
        # Shaping rewards
        if not action_success:
            reward -= 0.3
        if done and current_state[3] > 0 and (current_state[0] >= 95 and current_state[1] >= 95): # Ensure target zone for bonus
            reward += 100 * (current_state[4]/CARGO_CAPACITY)  # Delivery bonus
        if current_state[6] > 75:
            reward -= 0.2 * (current_state[6] - 75)  # Overheating penalty
            
        # Clamp state values
        final_state = (
            max(min(current_state[0], 100), 0),
            max(min(current_state[1], 100), 0),
            max(min(current_state[2], MAX_VELOCITY), 0),
            max(min(current_state[3], MAX_ENERGY), 0),
            max(min(current_state[4], CARGO_CAPACITY), 0),
            max(min(current_state[5], 100), 0),
            max(min(current_state[6], MAX_TEMP), -50)
        )
    
        return tuple(final_state), reward, done

    def train(self, num_episodes=1000):
        self.training = True
        for episode in range(num_episodes):
            current_episode_state = self._get_initial_state() # Use a different variable name
            episode_reward_sum = 0 # Use a different variable name
            
            while not self._is_terminal_state(current_episode_state):
                # Get action from RL engine
                action_tensor = self._select_action(current_episode_state)
                action_item = action_tensor.item() # Use a different variable name
                next_episode_state, current_reward, done_flag = self._take_action(current_episode_state, action_item) # Environment interaction
                loss = self._update_policy(current_episode_state, action_item, current_reward, next_episode_state) # Update policy
                
                # Track rewards
                episode_reward_sum += current_reward
                current_episode_state = next_episode_state
                if done_flag: break
    
            # Update exploration parameters
            self.rl_engine.epsilon *= self.rl_engine.epsilon_decay
            self.rl_engine.epsilon = max(
                self.rl_engine.min_epsilon, 
                self.rl_engine.epsilon
            )
    
            logger.info(f"Episode {episode} | Reward: {episode_reward_sum:.2f} | Loss: {loss if loss is not None else 'N/A':.4f}")
        self.training = False


    def _consolidate_memory(self):
        """Delegate memory consolidation to RL subsystem"""
        if hasattr(self.rl_engine, 'local_memory') and hasattr(self.rl_engine.local_memory, 'consolidate'):
            self.rl_engine.local_memory.consolidate()
        if hasattr(self.rl_engine, 'memory') and hasattr(self.rl_engine.memory, 'consolidate'):
            self.rl_engine.memory.consolidate()
        
        # Preserve existing memory logic
        if hasattr(super(), '_consolidate_memory'):
            super()._consolidate_memory()


    def _apply_alignment_corrections(self, report):
        """Adjust behavior based on alignment monitoring"""
        if report.get('fairness', {}).get('demographic_parity', 1.0) < 0.8: # Add default for fairness and demographic_parity
            self._adjust_policy_fairness()
        if report.get('value_alignment', 1.0) < 0.7: # Add default for value_alignment
            self._reroute_ethical_constraints()

    def _adjust_policy_fairness(self):
        """Placeholder for adjusting policy for fairness."""
        logger.info("Adjusting policy for fairness concerns.")
        # Example: Could involve re-weighting samples, adding constraints, or modifying reward function.

    def _reroute_ethical_constraints(self):
        """Placeholder for rerouting tasks based on ethical constraints."""
        logger.info("Rerouting task due to ethical constraint violation.")
        # Example: Could involve sending the task to a human reviewer or a specialized ethical AI.


    def evaluate(self, eval_episodes=10):
        """
        Evaluate the agent's performance without learning.

        """
        logger.info("Evaluating agent over %d episodes", eval_episodes)
        rewards = []
        
        for _ in range(eval_episodes):
            current_eval_state = self._get_initial_state() # Use a different variable name
            episode_reward_sum = 0 # Use a different variable name
            
            while not self._is_terminal_state(current_eval_state):
                action_tensor = self._select_action(current_eval_state, explore=False) # Renamed action
                action_item = action_tensor.item()
                next_eval_state, current_reward, done_flag = self._take_action(current_eval_state, action_item)  # Capture all three values
                episode_reward_sum += current_reward
                current_eval_state = next_eval_state
                if done_flag:  # Exit early if terminal state reached
                    break
            
            rewards.append(episode_reward_sum)
        
        avg_reward = np.mean(rewards) if rewards else 0.0 # Handle case where rewards list might be empty
        logger.info("Evaluation complete. Average reward: %.2f", avg_reward)
        return avg_reward
    
    def learn_from_demonstration(self, demonstration):
        """
        Learn from a provided demonstration.
        
        Args:
            demonstration (list): Sequence of (state, action, reward) tuples
        """
        logger.info("Learning from demonstration with %d steps", len(demonstration))
        
        for demo_state, demo_action, demo_reward in demonstration: # Renamed variables
            # Update policy towards demonstrated actions
            self._update_policy_from_demo(demo_state, demo_action)
            
            # Update value estimates
            self.value_estimates[demo_state] = (1 - self.learning_params['learning_rate']) * \
                                         self.value_estimates.get(demo_state, 0) + \
                                         self.learning_params['learning_rate'] * demo_reward
            
            # Store in memory
            self._store_experience(demo_state, demo_action, demo_reward)
        
        logger.info("Demonstration learning complete")
    
    def _select_action(self, state, explore=True):
        """Delegate action selection to RL engine"""
        state_tensor = torch.FloatTensor(self._extract_features(state))
        result = self.rl_engine.select_action(state_tensor, explore=explore)
        
        # Handle both tensor and tuple returns from RL engine
        if isinstance(result, tuple):
            # Assume first element is action tensor
            return result[0]
        return result

    def _compute_policy(self, state):
        state_features = self._extract_features(state)
        return self.policy.compute_policy(state_features)

    def _update_policy(self, state, action, reward, next_state):
        """Store experience and update RL policy"""
        # Convert to proper tensor format
        state_features = torch.FloatTensor(self._extract_features(state))
        next_features = torch.FloatTensor(self._extract_features(next_state))
        
        # Create transition
        current_transition = Transition( # Renamed variable
            state=state_features,
            action=torch.LongTensor([action]),
            reward=reward,
            next_state=next_features,
            done=self._is_terminal_state(next_state)
        )
        
        # Store in RL memory
        if hasattr(self.rl_engine, 'learner_memory') and hasattr(self.rl_engine.learner_memory, 'add'):
             self.rl_engine.learner_memory.add(current_transition)
        else:
            logger.warning("RL engine memory or add method not found. Skipping experience storage.")

        
        # Perform policy update
        if hasattr(self.rl_engine, 'update_policy'):
            return self.rl_engine.update_policy()
        else:
            logger.warning("RL engine update_policy method not found. Skipping policy update.")
            return None


    def _update_value_estimates(self, state, reward, next_state):
        """
        Update value estimates using TD learning.
        
        Args:
            state: Previous state
            reward: Reward received
            next_state: Resulting state
        """
        current_value = self.value_estimates.get(state, 0)
        next_value = self.value_estimates.get(next_state, 0)
        
        td_target = reward + self.learning_params['discount_factor'] * next_value
        self.value_estimates[state] = current_value + \
                                    self.learning_params['learning_rate'] * \
                                    (td_target - current_value)
    
    def _update_policy_from_demo(self, state, action):
        """
        Update policy towards demonstrated action.
        
        Args:
            state: Demonstrated state
            action: Demonstrated action
        """
        state_features = self._extract_features(state)
        
        # Check if policy_weights and its keys exist
        if not hasattr(self, 'policy_weights') or \
           'input_layer' not in self.policy_weights or \
           'hidden_layer' not in self.policy_weights or \
           'output_layer' not in self.policy_weights:
            logger.warning("Policy weights not properly initialized. Skipping update from demo.")
            return

        try:
            hidden1_activation = np.tanh(np.dot(state_features, self.policy_weights['input_layer']))
            hidden2_activation = np.tanh(np.dot(hidden1_activation, self.policy_weights['hidden_layer']))
            
            # Update towards demonstrated action
            target_probs = np.zeros(self.num_actions) # Use self.num_actions
            target_probs[action] = 1.0
            current_probs = self._compute_policy(state) # This computes based on self.policy (PolicyManager)
            
            # Gradient calculation needs to align with how PolicyManager's policy is structured/updated.
            # The direct weight update below assumes a simple 2-hidden-layer network with direct weight access,
            # which might differ from PolicyManager's PyTorch nn.Sequential model.
            # For simplicity, if PolicyManager has its own update method, prefer that.
            # If self.policy is a PolicyManager instance:
            td_error_for_demo = 1.0 # Assume demo action is good
            if hasattr(self.policy, 'update'):
                self.policy.update(state_features, action, td_error_for_demo, self.learning_params['learning_rate'])
            else:
                # Fallback to the original manual gradient update if PolicyManager does not have 'update'
                # This part needs careful review if PolicyManager's structure is complex.
                # The original code for grad was: grad = hidden[:, None] * (current - target)
                # This assumed `hidden` was the last hidden layer before output.
                # Let's use hidden2_activation.
                # Reshape hidden2_activation for broadcasting: (hidden_size,) -> (hidden_size, 1)
                error_signal = current_probs - target_probs # (action_dim,)
                # Gradient for output layer weights: hidden_activation (transpose) * error_signal
                # Output layer weights shape: (hidden2_size, action_dim)
                # hidden2_activation shape: (hidden2_size,)
                # error_signal shape: (action_dim,)
                grad_output_layer = np.outer(hidden2_activation, error_signal)
                self.policy_weights['output_layer'] -= self.learning_params['learning_rate'] * grad_output_layer
                # Backpropagate error for hidden_layer (simplified)
                # delta_hidden2 = np.dot(error_signal, self.policy_weights['output_layer'].T) * (1 - hidden2_activation**2) # Tanh derivative
                # grad_hidden_layer = np.outer(hidden1_activation, delta_hidden2)
                # self.policy_weights['hidden_layer'] -= self.learning_params['learning_rate'] * grad_hidden_layer
                # ... and so on for input_layer, this becomes complex manually.
                logger.debug("Updated policy from demo using PolicyManager or fallback manual update.")

        except Exception as e:
            logger.error(f"Error updating policy from demo: {e}")

    def _adapt_parameters(self, reward):
        """
        Adapt learning parameters based on recent performance.
        """
        # Track recent rewards
        self.last_reward = reward
        self.total_steps += 1
        
        # Decay exploration
        self.learning_params['exploration_rate'] *= 0.9995
        self.learning_params['exploration_rate'] = max(0.01, self.learning_params['exploration_rate'])
        
        # Adjust learning rate based on performance variance
        if len(self.performance_history) > 10:
            recent_perf = self.performance_history[-10:]
            perf_variance = np.var(recent_perf)
            
            if perf_variance < 0.1:  # Low variance, decrease learning rate
                self.learning_params['learning_rate'] *= 0.995
            elif perf_variance > 1.0:  # High variance, increase learning rate
                self.learning_params['learning_rate'] *= 1.01
            
            # Keep within bounds
            self.learning_params['learning_rate'] = np.clip(
                self.learning_params['learning_rate'], 1e-4, 0.1)
    
    def _consolidate_memory(self):
        """
        Consolidate recent experiences into long-term memory.
        """
        if hasattr(self.memory, 'consolidate'):
            self.memory.consolidate()

        if len(self.episodic_memory) > 0:
            recent_exp = self.episodic_memory[-1]  # Most recent experience
            state_key = f"state_{hash(recent_exp['state']) % 1000}"
            
            # Update semantic memory with important experiences
            if abs(recent_exp['reward']) > 0.8: # Check if reward exists and is significant
                current_semantic_value = self.semantic_memory.get(state_key, {}).get(recent_exp['action'], 0)
                self.semantic_memory[state_key][recent_exp['action']] = \
                    current_semantic_value * 0.9 + recent_exp['reward'] * 0.1

        # Buffer maintenance
        if hasattr(self.replay_buffer, '_remove_stale_experiences'):
            self.replay_buffer._remove_stale_experiences()

        # Update semantic memory with high-priority experiences
        if hasattr(self.replay_buffer, 'sample'):
            try:
                for exp in self.replay_buffer.sample(10, strategy='reward'):
                    if hasattr(exp, 'state') and hasattr(exp, 'action') and hasattr(exp, 'reward'):
                        if hasattr(self.semantic_memory, 'store'):
                             self.semantic_memory.store(exp.state, exp.action, exp.reward)
                        else: # Fallback to dict-like access if no 'store' method
                             key = f"state_{hash(exp.state) % 1000}"
                             self.semantic_memory[key][exp.action] = exp.reward
            except Exception as e:
                logger.warning(f"Error sampling from replay buffer or storing to semantic memory: {e}")


    def update_memory(self, key: str, value):
        """
        Update shared memory with a key-value pair.
        """
        self.shared_memory[key] = value # Assuming shared_memory is dict-like
    
    def retrieve_memory(self, key: str):
        """
        Retrieve value from shared memory.
        """
        return self.shared_memory.get(key, None)

    def _get_initial_state(self):
        """Generate an initial state with 7 elements as defined in _take_action"""
        return (
            random.uniform(0, 100),   # x_position
            random.uniform(0, 100),   # y_position
            random.uniform(0, 10),    # velocity
            random.uniform(50, 100),  # energy
            0,                        # cargo
            100,                      # integrity
            random.uniform(-50, 50)   # temperature
        )
    
    def _is_terminal_state(self, state):
        """Check if a state is terminal based on actual state components"""
        if not isinstance(state, tuple) or len(state) < 7: # Basic check for state structure
            logger.warning(f"Invalid state format for terminal check: {state}")
            return True # Treat invalid state as terminal to prevent errors
        return any([
            state[3] <= 0,        # No energy
            state[5] <= 0,        # System failure
            state[6] >= 100,      # Overheating
            (state[0] >= 95 and state[1] >= 95)  # Reached target zone
        ])
        
    def _extract_features(self, state):
        """Convert state into feature vector"""
        if not isinstance(state, tuple) or len(state) < 3: # Ensure state has at least 3 elements for basic calculations
            logger.warning(f"Invalid state format for feature extraction: {state}. Returning zero vector.")
            return np.zeros(self.state_dim) # Return a zero vector of appropriate dimension

        # Basic features from the first 3 elements, assuming state has at least 3 elements
        s0 = state[0] if len(state) > 0 else 0
        s1 = state[1] if len(state) > 1 else 0
        s2 = state[2] if len(state) > 2 else 0

        # Pad with zeros if state_dim is larger than what can be derived
        features = np.array([
            s0 / 10.0,
            s1 / 10.0,
            s2 / 10.0,
            sum(state[:3]) / 30.0 if len(state) >=3 else 0.0,
            min(state[:3]) / 10.0 if len(state) >=3 else 0.0,
            max(state[:3]) / 10.0 if len(state) >=3 else 0.0,
            len([s for s in state[:3] if s > 5]) / 3.0 if len(state) >=3 else 0.0,
            math.sqrt(sum(s**2 for s in state[:3])) / 17.32 if len(state) >=3 else 0.0, 
            (s0 * s1) / 100.0,
            (s1 * s2) / 100.0
        ])
        if len(features) < self.state_dim:
            features = np.pad(features, (0, self.state_dim - len(features)), 'constant')
        return features[:self.state_dim] # Ensure correct dimension

    def _basic_rl_skill(self, state):
        """Basic RL skill implementation"""
        action_tensor = self._select_action(state)
        action_item = action_tensor.item()
        next_state, current_reward, done_flag = self._take_action(state, action_item)
        self._update_policy(state, action_item, current_reward, next_state)
        return next_state, current_reward
    
    def _memory_skill(self, query):
        """Memory retrieval skill"""
        # Try MultiModalMemory first
        if hasattr(self.memory, 'retrieve'):
            results = self.memory.retrieve(query)
            if results: # Assuming retrieve returns a list, and first result is most relevant
                return results[0].get('data', results[0]) if isinstance(results[0], dict) else results[0]

        # Fallback to shared_memory (from BaseAgent)
        if self.shared_memory and query in self.shared_memory: # Check if shared_memory is not None
            return self.shared_memory[query]
        
        # Fallback to semantic_memory
        if query in self.semantic_memory:
            return self.semantic_memory[query]
        
        # Fallback: Search episodic memory (less efficient for direct query)
        for exp in reversed(self.episodic_memory):
            # A more robust check would be needed here, e.g., if query is a substring or matches a key
            if isinstance(exp, dict) and 'state' in exp and query in str(exp['state']):
                return exp
        return None # If query not found in any memory system

    def _routing_skill(self, message):
        try:
            if isinstance(message, dict) and "type" in message:
                if message["type"] == "train":
                    return self.execute({'type': 'train', 'episodes': message.get('episodes', 5)})
                elif message["type"] == "evaluate":
                    return self.execute({'type': 'evaluate', 'episodes': message.get('episodes', 10)}) # Added episodes for eval
                else:
                    return {"status": "unrecognized_message_type", "message_type": message["type"]}
            elif isinstance(message, str): # Handle simple string messages if needed
                 if "train" in message.lower():
                    return self.execute({'type': 'train', 'episodes': 5})
                 elif "evaluate" in message.lower():
                    return self.execute({'type': 'evaluate', 'episodes': 10})
                 else:
                    return {"status": "unrecognized_string_message", "content": message[:50]}
            else:
                return {"status": "unrecognized_message_format", "format": type(message).__name__}

        except Exception as e:
            error_type_name = type(e).__name__
            error_category = self.error_classes.get(error_type_name, 'unknown_error')
            logger.warning(f"Routing skill error ({error_category}): {str(e)}")
            return {"status": "error", "category": error_category, "message": str(e)}
        
    def _recover_soft_reset(self):
        """
        Soft recovery strategy.
        Clears recent episodic memory and resets minor tracking variables
        without touching weights or optimizer state.
        """
        logger.info("[Recovery] Performing soft reset: clearing episodic memory and resetting counters.")
        self.episodic_memory.clear()
        self.last_reward = 0
        self.total_steps = 0
        logger.info("[Recovery] Soft reset complete.")
        return True  # Indicate recovery succeeded
    
    def _recover_lr_adjustment(self):
        """
        Adjusts the learning rate up or down depending on recent performance.
        Helps escape local minima or stabilize learning.
        """
        logger.info("[Recovery] Adjusting learning rate for recovery.")
        prev_lr = self.learning_params['learning_rate']
        
        if len(self.performance_history) >= 10:
            recent_rewards = self.performance_history[-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0 # Handle empty list
            
            if avg_reward < 0:  # Poor performance → lower learning rate
                self.learning_params['learning_rate'] *= 0.9
                logger.info(f"[Recovery] Performance low; decreasing learning rate to {self.learning_params['learning_rate']:.6f}")
            else:  # Decent performance → cautiously increase
                self.learning_params['learning_rate'] *= 1.05
                logger.info(f"[Recovery] Performance acceptable; increasing learning rate to {self.learning_params['learning_rate']:.6f}")
            
            # Clamp within safe bounds
            self.learning_params['learning_rate'] = np.clip(self.learning_params['learning_rate'], 1e-5, 1e-1)
        else:
            logger.info("[Recovery] Not enough history; skipping learning rate adjustment.")
        
        return True
    
    def _recover_full_reset(self):
        """
        Full recovery: resets all weights, optimizer state, memory, and counters.
        Basically reinitializes the agent from scratch.
        """
        logger.info("[Recovery] Performing full reset: reinitializing policy, optimizer, and clearing all memory.")
        
        # Re-initialize PolicyManager (which includes the policy network)
        self.policy = PolicyManager(self.state_dim, self.num_actions)
        # policy_weights are managed within PolicyManager or its attached network, so direct reset of self.policy_weights might not be needed
        # if self.policy_weights was a separate attribute. If it was for a custom network, it would be reset here.
        
        # Reset routing policy and optimizer
        # self.state_dim and self.num_handlers should be correctly set from self.config
        self.routing_policy = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_handlers)
        )
        self.routing_optimizer = torch.optim.Adam(self.routing_policy.parameters())
        
        # Clear all memory
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.performance_history.clear()
        if hasattr(self.replay_buffer, 'clear'): # Check if replay buffer has a clear method
            self.replay_buffer.clear()
        
        # Reset counters and learning parameters to initial defaults
        self.last_reward = 0
        self.total_steps = 0
        self._init_subsystems() # Re-initialize learning_params and other components

        logger.info("[Recovery] Full reset complete.")
        return True

if __name__ == "__main__":
    print("\n=== Running Adaptive Agent ===\n")

    shared_memory = {}
    agent_factory = lambda name, cfg: None # Adjusted agent_factory placeholder

    config = {
        'state_dim': 10,
        'num_actions': 7, # Matching the 7 actions in _take_action
        'num_handlers': 3,
        'episodic_capacity': 1000,
        'experience_staleness_days': 1,
        'semantic_decay_rate': 0.95,
        'min_memory_strength': 0.05,
        'replay_capacity': 100000,
        'priority_alpha': 0.6,
        'per_beta': 0.4,
        'per_epsilon': 1e-6,
        'risk_threshold': 0.7,
        'retry_policy': {'max_attempts': 3, 'backoff_factor': 1.5},
        'replay_config': {
            'batch_size': 64,
            'per_beta': 0.6
        },
        'rl': { # Adding 'rl' sub-dictionary for consistency with how params are fetched
            'state_dim': 10,
            'num_actions': 7,
            'num_handlers': 3
        }
    }

    adaptive = AdaptiveAgent(shared_memory, agent_factory, config=config)

    print(adaptive)

    print("\n* * * * * Phase 2 * * * * *\n")

    state = (4, 2, 3, 100, 0, 100, 50)
    action = 1          # Valid action index
    reward = 2.5        # Numeric reward
    storage = adaptive._store_experience(state, action, reward)

    print(storage) # _store_experience doesn't return anything, so this will print None

    print("\n* * * * * Phase 3 * * * * *\n")

    next_s, rwd, dn = adaptive._take_action(state, action) # _take_action returns 3 values

    print(f"Next State: {next_s}\nReward: {rwd}\nDone: {dn}")

    print("\n* * * * * Phase 4 * * * * *\n")

    eval_episodes=5 # Reduced for quicker test

    eval_result = adaptive.evaluate(eval_episodes=eval_episodes)

    print(f"Average evaluation reward: {eval_result}")
    print("\n=== Successfully Ran Adaptive Agent ===\n")
