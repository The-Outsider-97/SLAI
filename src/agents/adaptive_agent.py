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

from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from src.agents.learning.learning_memory import LearningMemory
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
        self.episodic_memory = deque(maxlen=1000)  # Recent experiences
        self.semantic_memory = defaultdict(dict)   # Conceptual knowledge

        state_dim = self.config.get('state_dim', 10)
        num_actions = self.config.get('num_actions', 2)
        
        self.rl_engine = ReinforcementLearning(
            config=config,
            learning_memory=LearningMemory(config),
            multimodal_memory=MultiModalMemory(config)
        )

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

        self.task_scheduler = DeadlineAwareScheduler(
            risk_threshold=config.get('risk_threshold', 0.7),
            retry_policy=config.get('retry_policy', {'max_attempts': 3, 'backoff_factor': 1.5})
        )

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
            **({} if learning_params is None else learning_params)
        }

        self.memory = MultiModalMemory(config)
        self.tuner = LearningParameterTuner(learning_params)
        self.router = AdaptiveRouter(config)
        self.policy = PolicyManager(config['state_dim'], config['num_actions'])

        self.replay_buffer = DistributedReplayBuffer(
            user_config={
                'distributed': {
                    'capacity': config.get('replay_capacity', 100000),
                    'prioritization_alpha': config.get('priority_alpha', 0.6),
                    'staleness_threshold_days': config.get('experience_staleness_days', 1)
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
        state = list(state)
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
                if state[3] > 5 and state[6] < 80:
                    state[2] = min(state[2] + 1 + 0.5*stochasticity, MAX_VELOCITY)
                    state[3] -= 3 * (1 + state[2]/MAX_VELOCITY)
                    state[6] += 2
                    reward += state[2] * 0.2
                else:
                    action_success = False
                    reward -= 1
    
            elif action == 1:  # Decelerate
                state[2] = max(state[2] - 1 - stochasticity, 0)
                state[6] -= 1
                reward += state[2] * 0.1
    
            elif action == 2:  # Turn left
                new_y = state[1] + state[2] * (0.8 + stochasticity*0.4)
                if 0 <= new_y <= 100:
                    state[1] = new_y
                    reward += abs(state[2]) * 0.05
                else:
                    reward -= 2  # Wall collision
    
            elif action == 3:  # Turn right
                new_x = state[0] + state[2] * (0.8 + stochasticity*0.4)
                if 0 <= new_x <= 100:
                    state[0] = new_x
                    reward += abs(state[2]) * 0.05
                else:
                    reward -= 2  # Wall collision
    
            elif action == 4:  # Collect resource
                if state[4] < CARGO_CAPACITY and state[3] > 10:
                    collection = min(3 * (1 + stochasticity), CARGO_CAPACITY - state[4])
                    state[4] += collection
                    state[3] -= 5
                    reward += collection * 0.7
                    state[6] += 1
                else:
                    action_success = False
                    reward -= 0.5
    
            elif action == 5:  # Repair system
                if state[3] > 20:
                    repair_amount = 10 * (0.5 + stochasticity)
                    state[5] = min(state[5] + repair_amount, 100)
                    state[3] -= 15
                    reward += repair_amount * 0.3
                    state[6] += 2
                else:
                    action_success = False
    
            elif action == 6:  # Cool system
                cooling = 5 * (0.8 + stochasticity*0.4)
                state[6] = max(state[6] - cooling, -50)
                state[3] -= 2
                reward += cooling * 0.2
    
            # Environmental dynamics
            state[3] = max(state[3] - 0.1 * state[2], 0)  # Energy drain
            state[6] += 0.05 * state[2]**2  # Friction heating
            if random.random() < 0.1:  # Random environmental events
                state[6] += 5 * random.random()
                
        except Exception as e:
            logger.error(f"Action execution error: {str(e)}")
            reward -= 5
            action_success = False
    
        # Terminal conditions
        done = any([
            state[3] <= 0,  # No energy
            state[5] <= 0,  # System failure
            state[6] >= MAX_TEMP,  # Overheating
            (state[0] >= 95 and state[1] >= 95)  # Reached target zone
        ])
        
        # Shaping rewards
        if not action_success:
            reward -= 0.3
        if done and state[3] > 0:
            reward += 100 * (state[4]/CARGO_CAPACITY)  # Delivery bonus
        if state[6] > 75:
            reward -= 0.2 * (state[6] - 75)  # Overheating penalty
            
        # Clamp state values
        state = (
            max(min(state[0], 100), 0),
            max(min(state[1], 100), 0),
            max(min(state[2], MAX_VELOCITY), 0),
            max(min(state[3], MAX_ENERGY), 0),
            max(min(state[4], CARGO_CAPACITY), 0),
            max(min(state[5], 100), 0),
            max(min(state[6], MAX_TEMP), -50)
        )
    
        return tuple(state), reward, done

    def train(self, num_episodes=1000):
        self.training = True
        for episode in range(num_episodes):
            state = self._get_initial_state()
            episode_reward = 0
            
            while not self._is_terminal_state(state):
                # Get action from RL engine
                action_tensor = self._select_action(state)
                action = action_tensor.item()
                next_state, reward, done = self._take_action(state, action) # Environment interaction
                loss = self._update_policy(state, action, reward, next_state) # Update policy
                
                # Track rewards
                episode_reward += reward
                state = next_state
    
            # Update exploration parameters
            self.rl_engine.epsilon *= self.rl_engine.epsilon_decay
            self.rl_engine.epsilon = max(
                self.rl_engine.min_epsilon, 
                self.rl_engine.epsilon
            )
    
            logger.info(f"Episode {episode} | Reward: {episode_reward:.2f} | Loss: {loss:.4f}")

    def _consolidate_memory(self):
        """Delegate memory consolidation to RL subsystem"""
        self.rl_engine.local_memory.consolidate()
        self.rl_engine.memory.consolidate()
        
        # Preserve existing memory logic
        super()._consolidate_memory()

    def _apply_alignment_corrections(self, report):
        """Adjust behavior based on alignment monitoring"""
        if report['fairness']['demographic_parity'] < 0.8:
            self._adjust_policy_fairness()
        if report['value_alignment'] < 0.7:
            self._reroute_ethical_constraints()

    def evaluate(self, eval_episodes=10):
        """
        Evaluate the agent's performance without learning.

        """
        logger.info("Evaluating agent over %d episodes", eval_episodes)
        rewards = []
        
        for _ in range(eval_episodes):
            state = self._get_initial_state()
            episode_reward = 0
            
            while not self._is_terminal_state(state):
                action = self._select_action(state, explore=False)
                next_state, reward, done = self._take_action(state, action)  # Capture all three values
                episode_reward += reward
                state = next_state
                if done:  # Exit early if terminal state reached
                    break
            
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
        """Delegate action selection to RL engine"""
        state_tensor = torch.FloatTensor(self._extract_features(state))
        return self.rl_engine.select_action(state_tensor, explore=explore)

    def _compute_policy(self, state):
        state_features = self._extract_features(state)
        return self.policy.compute_policy(state_features)

    def _update_policy(self, state, action, reward, next_state):
        """Store experience and update RL policy"""
        # Convert to proper tensor format
        state_features = torch.FloatTensor(self._extract_features(state))
        next_features = torch.FloatTensor(self._extract_features(next_state))
        
        # Create transition
        transition = Transition(
            state=state_features,
            action=torch.LongTensor([action]),
            reward=reward,
            next_state=next_features,
            done=self._is_terminal_state(next_state)
        )
        
        # Store in RL memory
        self.rl_engine.learner_memory.add(transition)
        
        # Perform policy update
        return self.rl_engine.update_policy()

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
        return any([
            state[3] <= 0,        # No energy
            state[5] <= 0,        # System failure
            state[6] >= 100,      # Overheating
            (state[0] >= 95 and state[1] >= 95)  # Reached target zone
        ])
        
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
        'retry_policy': {'max_attempts': 3, 'backoff_factor': 1.5},
        'replay_config': {  # Add replay configuration
            'batch_size': 64,
            'per_beta': 0.6
        }
    }

    adaptive = AdaptiveAgent(shared_memory, agent_factory, config=config)

    print(adaptive)

    print("\n* * * * * Phase 2 * * * * *\n")

    state = (4, 2, 3, 100, 0, 100, 50)
    action = 1          # Valid action index
    reward = 2.5        # Numeric reward
    storage = adaptive._store_experience(state, action, reward)

    print(storage)

    print("\n* * * * * Phase 3 * * * * *\n")

    action = adaptive._take_action(state, action)

    print(action)

    print("\n* * * * * Phase 4 * * * * *\n")

    eval_episodes=20

    eval = adaptive.evaluate(eval_episodes=eval_episodes)

    print(eval)
    print("\n=== Successfully Ran Adaptive Agent ===\n")
