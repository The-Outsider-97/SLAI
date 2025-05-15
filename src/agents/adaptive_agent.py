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
import time as timedelta
import torch.nn.functional as F
import statsmodels.formula.api as smf

from collections import defaultdict, deque

from src.agents.alignment.alignment_monitor import AlignmentMonitor
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from src.agents.adaptive.policy_manager import PolicyManager
from src.agents.adaptive.parameter_tuner import LearningParameterTuner
from src.agents.adaptive.memory_system import MultiModalMemory
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger

logger = get_logger("Adaptive Agent")

CONFIG_PATH = "src/agents/alignment/configs/alignment_config.yaml"

class AdaptiveAgent(BaseAgent):
    """
    An adaptive agent that combines reinforcement learning with memory and routing capabilities.
    Continuously improves from feedback, success/failure, and demonstrations.
    """
    
    def __init__(self, shared_memory, agent_factory, config=None, learning_params=None, args=(), kwargs={}):
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
        self.episodic_memory = deque(maxlen=1000)  # Recent experiences
        self.semantic_memory = defaultdict(dict)   # Conceptual knowledge
        self.learning_params = {
            'learning_rate': 0.01,
            'exploration_rate': 0.3,
            'discount_factor': 0.95,
            'temperature': 1.0,  # For softmax decision making
            'memory_capacity': 1000,
            **({} if learning_params is None else learning_params)
        }
        state_dim = self.config.get('state_dim', 10)
        num_actions = self.config.get('num_actions', 2)
        self.policy = PolicyManager(state_dim, num_actions)
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
        state_dim = self.config.get('state_dim', 10)
        num_handlers = self.config.get('num_handlers', 3)

        # Core subsystems
        from src.collaborative.task_router import AdaptiveRouter
        self.memory = MultiModalMemory(config)
        self.tuner = LearningParameterTuner(learning_params)
        self.router = AdaptiveRouter(config)
        self.policy = PolicyManager(config['state_dim'], config['num_actions'])
        self.replay = ExperienceReplayManager(
            self.memory.replay_buffer, 
            self.policy.network
        )

        self.routing_policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_handlers)
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

        # Alignment implementation
        self.alignment_monitor = AlignmentMonitor(
            sensitive_attributes=['gender', 'age_group'],
            config=CONFIG_PATH(
                fairness_metrics=['demographic_parity', 'equal_opportunity'],
                drift_threshold=0.1
            )
        )

        self.task_scheduler = DeadlineAwareScheduler(
            risk_threshold=config.get('risk_threshold', 0.7),
            retry_policy=config.get('retry_policy', {'max_attempts': 3, 'backoff_factor': 1.5})
        )

        from src.utils.replay_buffer import DistributedReplayBuffer, ExperienceReplayManager
        self.replay_buffer = DistributedReplayBuffer(
            capacity=config.get('replay_capacity', 100000),
            prioritization_alpha=config.get('priority_alpha', 0.6),
            staleness_threshold=timedelta(
                days=config.get('experience_staleness_days', 1)
            )
        )

        # Add priority experience replay parameters
        self.per_beta = config.get('per_beta', 0.4)
        self.per_epsilon = config.get('per_epsilon', 1e-6)

        
        logger.info("AdaptiveAgent initialized with parameters: %s", self.learning_params)

    def _store_experience(self, state, action, reward):
        """Store experience in prioritized replay buffer"""
        self.memory.store_experience(state, action, reward)
        next_state = self._extract_routing_features(state)  # Or get actual next state
        self.replay_buffer.push(
            agent_id=self.agent_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False
        )

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
        Simulate taking an action in the environment.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            tuple: (next_state, reward)
        """
        # Simple environment dynamics
        if action == 0:  # Action 0
            next_state = tuple(s + 1 for s in state)
            reward = -0.1 + random.random() * 0.2
        else:  # Action 1
            next_state = tuple(s * 1.5 for s in state)
            reward = sum(state) / 10 + random.random() * 0.5

        done = self._is_terminal_state(next_state)
        return next_state, reward, done

    def train(self, num_episodes=1000):
        self.training = True
        for episode in range(num_episodes):
            self.episode_reward = 0.0
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self._update_model(state, action, reward, next_state)
                self.episode_reward += reward
                state = next_state

            # Update parameter tuner with recent performance
            self.recent_rewards.append(self.episode_reward)
            self.tuner.update_performance(self.episode_reward)
            self.tuner.adapt(list(self.recent_rewards)[-10:])  # Last 10 episodes
            self.tuner.decay_exploration()

            # Apply tuned parameters
            self.learning_rate = self.tuner.params['learning_rate']
            self.exploration_rate = self.tuner.params['exploration_rate']

            logger.info(f"Episode {episode} Reward: {self.episode_reward:.2f} "
                       f"LR: {self.learning_rate:.4f} "
                       f"Exploration: {self.exploration_rate:.2f}")

        self.training = False

    def _apply_alignment_corrections(self, report):
        """Adjust behavior based on alignment monitoring"""
        if report['fairness']['demographic_parity'] < 0.8:
            self._adjust_policy_fairness()
        if report['value_alignment'] < 0.7:
            self._reroute_ethical_constraints()

    def evaluate(self, eval_episodes=10):
        """
        Evaluate the agent's performance without learning.
        
        Args:
            eval_episodes (int): Number of evaluation episodes
            
        Returns:
            float: Average reward across episodes
        """
        logger.info("Evaluating agent over %d episodes", eval_episodes)
        rewards = []
        
        for _ in range(eval_episodes):
            state = self._get_initial_state()
            episode_reward = 0
            
            while not self._is_terminal_state(state):
                action = self._select_action(state, explore=False)
                state, reward = self._take_action(state, action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        logger.info("Evaluation complete. Average reward: %.2f", avg_reward)
        return avg_reward
    
    def learn_from_demonstration(self, demonstration):
        """
        Learn from a provided demonstration.
        
        Args:
            demonstration (list): Sequence of (state, action, reward) tuples
        """
        logger.info("Learning from demonstration with %d steps", len(demonstration))
        
        for state, action, reward in demonstration:
            # Update policy towards demonstrated actions
            self._update_policy_from_demo(state, action)
            
            # Update value estimates
            self.value_estimates[state] = (1 - self.learning_params['learning_rate']) * \
                                         self.value_estimates.get(state, 0) + \
                                         self.learning_params['learning_rate'] * reward
            
            # Store in memory
            self._store_experience(state, action, reward)
        
        logger.info("Demonstration learning complete")
    
    def _select_action(self, state, explore=True):
        state_features = self._extract_features(state)
        return self.policy.get_action(state_features, explore=explore)

    def _compute_policy(self, state):
        state_features = self._extract_features(state)
        return self.policy.compute_policy(state_features)

    def _update_policy(self, state, action, reward, next_state):
        """Policy update with TD error calculation for prioritization"""
        # Get current and next state values
        current_value = self.value_estimates.get(state, 0)
        next_value = self.value_estimates.get(next_state, 0)
        
        # Calculate TD error with discount factor
        td_target = reward + self.learning_params['discount_factor'] * next_value
        td_error = td_target - current_value
        
        # Get policy gradient (existing logic)
        state_features = self._extract_features(state)
        hidden = np.tanh(np.dot(state_features, self.policy_weights['input_layer']))
        hidden = np.tanh(np.dot(hidden, self.policy_weights['hidden_layer']))
        grad = hidden * (1 if action == 0 else -1)  # Simplified gradient
        
        # Update weights with learning rate
        update = self.learning_params['learning_rate'] * td_error * grad
        self.policy_weights['output_layer'] += update
        
        # Store experience in buffer (now handled by replay buffer)
        self._store_experience(state, action, reward)
        
        return abs(td_error) + 1e-5

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
        hidden = np.tanh(np.dot(state_features, self.policy_weights['input_layer']))
        hidden = np.tanh(np.dot(hidden, self.policy_weights['hidden_layer']))
        
        # Update towards demonstrated action
        target = np.zeros(2)
        target[action] = 1.0
        current = self._compute_policy(state)
        
        # Cross-entropy gradient
        grad = hidden[:, None] * (current - target)
        self.policy_weights['output_layer'] -= self.learning_params['learning_rate'] * grad
    
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
    
    def _store_experience(self, state, action, reward):
        """
        Store experience in memory systems.
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': time.time()
        }
        
        self.episodic_memory.append(experience)
        
        # Also store in semantic memory if significant
        if abs(reward) > 1.0:
            key = f"state_{hash(state) % 1000}"
            self.semantic_memory[key][action] = reward
    
    def _consolidate_memory(self):
        """
        Consolidate recent experiences into long-term memory.
        """
        self.memory.consolidate()
        if len(self.episodic_memory) > 0:
            recent_exp = self.episodic_memory[-1]  # Most recent experience
            state_key = f"state_{hash(recent_exp['state']) % 1000}"
            
            # Update semantic memory with important experiences
            if abs(recent_exp['reward']) > 0.8:
                self.semantic_memory[state_key][recent_exp['action']] = \
                    self.semantic_memory[state_key].get(recent_exp['action'], 0) * 0.9 + \
                    recent_exp['reward'] * 0.1

        # Buffer maintenance
        self.replay_buffer._remove_stale_experiences()

        # Update semantic memory with high-priority experiences
        for exp in self.replay_buffer.sample(10, strategy='reward'):
            self.semantic_memory.store(exp.state, exp.action, exp.reward)

    def update_memory(self, key: str, value):
        """
        Update shared memory with a key-value pair.
        """
        self.shared_memory[key] = value
    
    def retrieve_memory(self, key: str):
        """
        Retrieve value from shared memory.
        """
        return self.shared_memory.get(key, None)

    def _get_initial_state(self):
        """Generate an initial state for an episode"""
        return tuple(np.random.randint(0, 10, size=3))
    
    def _is_terminal_state(self, state):
        """Check if a state is terminal"""
        return sum(state) > 25  # Simple terminal condition
        
    def _extract_features(self, state):
        """Convert state into feature vector"""
        return np.array([
            state[0] / 10.0,
            state[1] / 10.0,
            state[2] / 10.0,
            sum(state) / 30.0,
            min(state) / 10.0,
            max(state) / 10.0,
            len([s for s in state if s > 5]) / 3.0,
            math.sqrt(sum(s**2 for s in state)) / 17.32,  # sqrt(300)
            (state[0] * state[1]) / 100.0,
            (state[1] * state[2]) / 100.0
        ])
    
    def _basic_rl_skill(self, state):
        """Basic RL skill implementation"""
        action = self._select_action(state)
        next_state, reward = self._take_action(state, action)
        self._update_policy(state, action, reward, next_state)
        return next_state, reward
    
    def _memory_skill(self, query):
        """Memory retrieval skill"""
        results = self.memory.retrieve(query)
        if query in self.shared_memory:
            return self.shared_memory[query]
        elif query in self.semantic_memory:
            return self.semantic_memory[query]
        else:
            # Search episodic memory
            for exp in reversed(self.episodic_memory):
                if query in str(exp['state']):
                    return exp

        return results[0]['data'] if results else None
    
    def _routing_skill(self, message):
        try:
            if "train" in message:
                return self.execute({'type': 'train', 'episodes': 5})
            elif "evaluate" in message:
                return self.execute({'type': 'evaluate'})
            else:
                return {"status": "unrecognized_message"}
        except Exception as e:
            error_type = type(e).__name__
            error_category = self.error_classes.get(error_type, 'unknown_error')
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
            avg_reward = np.mean(recent_rewards)
            
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
        logger.info("[Recovery] Performing full reset: reinitializing weights, optimizer, and clearing all memory.")
        
        # Reset policy weights
        self.policy_weights = self._initialize_weights()
        
        # Reset routing policy and optimizer
        state_dim = self.config.get('state_dim', 10)
        num_handlers = self.config.get('num_handlers', 3)
        self.routing_policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_handlers)
        )
        self.routing_optimizer = torch.optim.Adam(self.routing_policy.parameters())
        
        # Clear all memory
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.performance_history.clear()
        
        # Reset counters
        self.last_reward = 0
        self.total_steps = 0
        
        logger.info("[Recovery] Full reset complete.")
        return True

if __name__ == "__main__":
    print("\n=== Running Adaptive Agent ===\n")

    shared_memory = {}
    agent_factory = lambda: None

    config = {
        'state_dim': 10,
        'num_actions': 2,
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
        'retry_policy': {'max_attempts': 3, 'backoff_factor': 1.5}
    }

    adaptive = AdaptiveAgent(shared_memory, agent_factory, config=config, learning_params=None)

    print(adaptive)
    print("\n=== Successfully Ran Adaptive Agent ===\n")
