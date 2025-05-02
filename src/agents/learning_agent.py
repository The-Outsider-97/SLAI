"""
SLAI Learning Agent: Core Component for Autonomous Learning & Improvement

Academic References:
1. DQN & RL: Mnih et al. (2015). Human-level control through deep RL. Nature.
2. MAML: Finn et al. (2017). Model-Agnostic Meta-Learning. PMLR.
3. RSI: Schmidhuber (2013). PowerPlay: Training General Problem Solvers.
4. Continual Learning: Parisi et al. (2019). Continual Learning Survey. IEEE TCDS.

Academic Foundations:
- Catastrophic Forgetting: Kirkpatrick et al. (2017) EWC
- Concept Drift: Gama et al. (2014) Survey on Concept Drift
- Meta-Learning: Finn et al. (2017) MAML
- Evolutionary Strategies: Salimans et al. (2017)
"""

import torch
import time
import psutil
import random
import logging
import os, sys
import warnings
import functools
import numpy as np
import torch.nn as nn
import gymnasium as gym

from collections import deque, defaultdict, OrderedDict
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from functools import partial

from src.agents.learning.dqn import DQNAgent
from src.agents.learning.maml_rl import MAMLAgent
from src.agents.learning.rsi import RSI_Agent
from src.agents.learning.rl_agent import RLAgent
from src.agents.safety_agent import SafeAI_Agent
from src.agents.base_agent import BaseAgent, LightMetricStore, LazyAgent


logger = logging.getLogger(__name__)

def validation_logic(self):
    required_params = {
        'dqn': ['hidden_size', 'gamma'],
        'maml': ['meta_lr', 'inner_lr']
    }
    for agent_type, params in required_params.items():
        if not all(k in self.config[agent_type] for k in params):
            raise ValueError(f"Missing params for {agent_type}: {params}")

class NaNException(Exception):
    """Raised when a NaN is encountered during learning"""
    def __init__(self, message="NaN value detected in training"):
        super().__init__(message)

class GradientExplosionError(Exception):
    """Raised when gradient norms exceed a safety threshold"""
    def __init__(self, norm, threshold=1e3):
        super().__init__(f"Gradient explosion detected: norm={norm:.2f}, threshold={threshold}")

class InvalidActionError(Exception):
    """Raised when an action fails safety validation or is undefined"""
    def __init__(self, action=None):
        message = f"Invalid or unsafe action: {action}" if action else "Invalid or unsafe action encountered"
        super().__init__(message)

class InvalidConfigError(Exception):
    """Raised when agent configuration validation fails"""
    def __init__(self, message="Invalid agent configuration"):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f'InvalidConfigError: {self.message}'

class MultiTaskLearner(nn.Module):
    def __init__(self, shared_dim=256, task_dims=None):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU()
        )
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(shared_dim, dim) 
            for task, dim in task_dims.items()
        })

    def forward(self, x, task):
        shared = self.shared_encoder(x)
        return self.task_heads[task](shared)
    
class LearningAgent(BaseAgent):
    """Orchestrates SLAI's lifelong learning capabilities through multiple strategies"""

    def __init__(self, shared_memory, agent_factory, slai_lm, safety_agent, env=None, config=None):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        """
        Initialize learning subsystems with environment context
        
        Args:
            env: OpenAI-like environment
            config: Dictionary with agent configurations
        """
        self.slai_lm = slai_lm
        self.env = env
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.safety_agent = safety_agent
        self.config = config or {}
        self.rl_algorithm = self.config.get("rl_algorithm", None)
        self.strategy_weights = np.array([0.25]*4)  # [RL, DQN, MAML, RSI]
        self.prediction_weights = self.config.get('prediction_weights', [0.25, 0.25, 0.25, 0.25])
        self.performance_history = deque(maxlen=1000)

        self.embedding_buffer = deque(maxlen=512)  # store latest 512 embeddings

        self.policy_net = nn.Sequential(
            nn.Linear(512, 256),  # input_dim (embeddings) → hidden_dim
            nn.ReLU(),
            nn.Linear(256, 2)    # hidden_dim → num_classes (binary: 0 or 1)
        )
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def performance_metrics(self):
        return self._performance_metrics
    
    @performance_metrics.setter
    def performance_metrics(self, value):
        self._performance_metrics = value
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss

        # Initialize learning subsystems
        self._initialize_agents(
            env=self.env,
            performance_metrics=self.performance_metrics,
            shared_memory=self.shared_memory
        )
        if self.env is None or not hasattr(self.env, 'observation_space') or not hasattr(self.env, 'action_space'):
            raise ValueError("LearningAgent requires a valid environment with observation_space and action_space.")
        else:
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n

        possible_actions = list(range(action_dim))

        self.agents = {
            'dqn': DQNAgent(state_dim, action_dim, self.config.get('dqn', {})),
            'maml': MAMLAgent(state_dim, action_dim, **self.config.get('maml', {})),
            'rsi': RSI_Agent(state_dim, action_dim, self.shared_memory, config=self.config.get('rsi', {})),
            'rl': RLAgent(possible_actions, **self.config.get('rl', {}))
        }
        
        # Enhanced MAML initialization
        self.maml_task_pool_size = self.config.get('maml_task_pool_size', 100)
        self.maml_task_pool = deque(maxlen=self.maml_task_pool_size)
        
        # Deeper RSI integration
        self.rsi_improvement_cycle = self.config.get('rsi_improvement_cycle', 50)
        self.architecture_history = deque(maxlen=10)  # Tracks network changes
        
        # Integrated strategy evaluator
        self.strategy_evaluator = MetaStrategyEvaluator(
            agents=self.agents,
            performance_metrics=self.performance_metrics
        )

        self._setup_continual_learning()
        self._setup_trigger_system()

        # State tracking
        self.concept_drift_detector = ConceptDriftDetector()
        self.last_retraining = datetime.now()
        self.slai_lm = slai_lm

        # Error recovery state tracking
        self.error_history = deque(maxlen=100)
        self.consecutive_errors = 0
        self.recovery_strategies = [
            self._recover_soft_reset,
            self._recover_learning_rate_adjustment,
            self._recover_strategy_switch,
            self._recover_full_reset
        ]

        # Defer heavy initialization
        self._deferred_initialization()

        logger.info(f"[TIME] create executed in {time.time()-start_time:.2f} seconds")
        logger.info(f"[MEMORY] create used {(psutil.Process().memory_info().rss - mem_before)/1024/1024:.2f} MB")

    def _deferred_initialization(self):
        """Initialize heavy components only when needed"""
        # Lightweight agent shells
        self.agents = {
            'dqn': LazyAgent(partial(self._create_dqn_agent)),
            'maml': LazyAgent(partial(self._create_maml_agent)),
            'rsi': LazyAgent(partial(self._create_rsi_agent)),
            'rl': LazyAgent(partial(self._create_rl_agent))
        }

        # Shared components
        self.memory = BaseAgent.SharedMemoryView(self.shared_memory)
        self.performance_metrics = LightMetricStore()
        
        # Configuration with memory limits
        self._config = {
            'max_network_size': 256,  # Hidden units
            'max_task_pool': 50,
            'max_history': 500
        }

    def _create_dqn_agent(self):
        """Create DQN agent with optimized network"""
        return DQNAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            config={
                'hidden_size': min(128, self._config['max_network_size']),
                'buffer_size': 2000,
                'batch_size': 32
            }
        )

    def _create_maml_agent(self):
        """Create MAML agent with shared network components"""
        return MAMLAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            hidden_size=min(64, self._config['max_network_size']),
            shared_components=self.shared_memory.get('shared_networks')
        )

    def _create_rsi_agent(self):
        """Create RSI agent with memory limits"""
        return RSI_Agent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            shared_memory=self.memory,
            config={'memory_size': 1000}
        )

    def _create_rl_agent(self):
        """Create basic RL agent"""
        return RLAgent(
            possible_actions=list(range(self.env.action_space.n)),
            learning_rate=0.001,
            discount_factor=0.95
        )

    def observe(self, embedding_with_label):
        """Store live embeddings into buffer."""
        if isinstance(embedding_with_label, np.ndarray):
            embedding_with_label = torch.tensor(embedding_with_label, dtype=torch.float32)
        if embedding_with_label.dim() == 2:
            embedding_with_label = embedding_with_label.squeeze(0)
    
        embedding = embedding_with_label[:-1]  # all but last value
        label = embedding_with_label[-1].long()  # last value as integer label (0 or 1)
    
        self.embedding_buffer.append((embedding, label))
    
    def train_from_embeddings(self):
        """Train a small supervised task on buffered embeddings."""
        if not hasattr(self, 'embedding_buffer') or len(self.embedding_buffer) < 16:
            return  # Not enough data yet
    
        batch = list(self.embedding_buffer)  # Grab batch
        embeddings = torch.stack([pair[0] for pair in batch])  # extract embeddings
        labels = torch.tensor([pair[1] for pair in batch], dtype=torch.long)  # extract labels
    
        # Move to tensor
        #inputs = torch.FloatTensor(inputs)
        inputs = embeddings
    
        # Simulate prediction targets
        targets = torch.randint(0, 2, (inputs.shape[0],))  # Random 0/1 labels
        targets = targets.long()
    
        # Forward pass into the network and loss
        preds = self.policy_net(embeddings)
        loss = self.loss_fn(preds, labels)
    
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        logger.info(f"[LearningAgent] Trained on {len(batch)} embeddings. Loss: {loss.item():.4f}")
        self.embedding_buffer.clear()

    def update_from_embeddings(self, inputs, targets):
        """Train directly from embeddings without full transition tuples."""
        if inputs.shape[0] == 0:
            return 0.0
    
        preds = self.policy_net.forward(inputs)
        loss = np.mean((preds - targets)**2)  # Simple MSE
    
        self.policy_net.backward(inputs, targets, learning_rate=self.lr)
    
        return loss

    def _create_task_variation(self):
        import copy
        """Enhanced environment variation generator for MAML"""
        try:
            env_variant = copy.deepcopy(self.env)
            
            # Systematic parameter randomization
            variation_params = {}
            if hasattr(env_variant.unwrapped, 'dynamics_config'):
                # Physics-based environments
                dynamics = env_variant.unwrapped.dynamics_config
                variation_params = {
                    'mass': np.clip(dynamics.mass * np.random.uniform(0.3, 3)), 
                    'damping': dynamics.damping * np.random.uniform(0.5, 2),
                    'gravity': np.clip(dynamics.gravity * np.random.uniform(0.5, 1.5)), 
                }
                env_variant.unwrapped.configure(**variation_params)
            
            # Reward shaping variations
            if hasattr(env_variant.unwrapped, 'reward_weights'):
                original_weights = env_variant.unwrapped.reward_weights
                variation_params['reward_weights'] = {
                    k: v * np.random.uniform(0.7, 1.3) 
                    for k,v in original_weights.items()
                }
                env_variant.unwrapped.reward_weights = variation_params['reward_weights']
            
            # Add to task pool for MAML
            self.maml_task_pool.append(env_variant)
            return env_variant
            
        except Exception as e:
            self.logger.error(f"Task variation failed: {str(e)}")
            return self.env  # Fallback to base env

    def _calculate_reward(self, state, action, next_state):
        base_reward = self.env.get_reward()
        
        # Safety penalties
        safety_penalty = 0
        if self.safety_agent.is_unsafe(action):
            safety_penalty += 2.0
            
        # Alignment penalties
        alignment_report = self.alignment_monitor.get_latest_report()
        alignment_penalty = 1.0 - alignment_report['ethics_score']
        
        # Curiosity bonus
        curiosity_bonus = self._calculate_novelty_bonus(state)
        
        return (base_reward 
                - safety_penalty 
                - alignment_penalty 
                + 0.3 * curiosity_bonus)

    def _calculate_novelty_bonus(self, state):
        """Intrinsic motivation for exploration"""
        return 1.0 / (1.0 + self.memory.count_similar(state))

    def _run_rsi_self_improvement(self):
        """Enhanced RSI integration with architectural evolution"""
        analysis = self.agents["rsi"].execute({
            "performance_history": self.performance_history,
            "strategy_weights": self.strategy_weights,
            "network_metrics": self._get_network_metrics()
        })
        
        # Apply architectural improvements
        if 'network_architecture' in analysis:
            new_arch = analysis['network_architecture']
            self._evolve_network_architecture(new_arch)
            self.architecture_history.append(new_arch)
        
        # Strategy reweighting based on RSI analysis
        if 'strategy_weights' in analysis:
            self.strategy_weights = np.clip(
                analysis['strategy_weights'], 0.1, 0.8
            )
            self.logger.info(f"RSI updated strategy weights: {self.strategy_weights}")

    def _evolve_network_architecture(self, new_arch):
        """Dynamic network architecture modification"""
        for agent_id in ['dqn', 'maml']:
            agent = self.agents[agent_id]
            if hasattr(agent, 'policy_net'):
                self.logger.info(f"Evolving {agent_id} network architecture")
                new_net = self._build_network_from_arch(new_arch)
                agent.policy_net = new_net
                agent.target_net.load_state_dict(new_net.state_dict())

    def _get_network_metrics(self):
        """Collect network health metrics for RSI analysis"""
        return {
            agent_id: {
                'gradient_norms': self._get_gradient_norm(agent),
                'activation_stats': self._get_activation_stats(agent),
                'parameter_count': sum(p.numel() for p in agent.policy_net.parameters())
            }
            for agent_id, agent in self.agents.items()
            if hasattr(agent, 'policy_net')
        }

    def meta_learn(self, num_tasks=5):
        """Enhanced MAML training with real task variations"""
        if len(self.maml_task_pool) < num_tasks:
            self.logger.warning("Insufficient tasks for MAML training")
            return
        
        # Select diverse tasks from pool
        task_variants = random.sample(self.maml_task_pool, num_tasks)
        task_batch = [(env, self._collect_adaptation_data(env)) for env in task_variants]
        
        # Perform meta-update
        meta_loss = self.agents["maml"].meta_update(task_batch)
        self.performance_metrics['meta_loss'].append(meta_loss)
        
        # Update strategy weights based on meta-learning performance
        self.strategy_weights[2] *= (1 + 1/(meta_loss + 1e-6))  # MAML index is 2

    def _collect_adaptation_data(self, env):
        """Collect adaptation data for a specific task variant"""
        state = env.reset()
        episode_data = []
        for _ in range(self.config.get('maml_adaptation_steps', 10)):
            action = self.agents["maml"].get_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward, next_state, done))
            state = next_state if not done else env.reset()
        return episode_data

    def _validate_config(self, config):
        """Centralized configuration validation"""
        required_params = {
            'dqn': ['hidden_size', 'gamma', 'epsilon_decay', 'buffer_size'],
            'maml': ['meta_lr', 'inner_lr', 'adaptation_steps'],
            'rsi': ['memory_size', 'exploration_rate', 'plasticity'],
            'rl': ['learning_rate', 'discount_factor']
        }
        
        error_messages = []
        for agent_type, params in required_params.items():
            if agent_type not in config:
                error_messages.append(f"Missing {agent_type} config section")
                continue
                
            missing = [p for p in params if p not in config[agent_type]]
            if missing:
                error_messages.append(
                    f"Missing in {agent_type} config: {', '.join(missing)}"
                )
        
        if error_messages:
            raise InvalidConfigError(
                "Configuration validation failed:\n- " + "\n- ".join(error_messages)
            )
        
        # Type validation
        if not isinstance(config.get('dqn', {}).get('hidden_size'), int):
            error_messages.append("DQN hidden_size must be integer")
        
        if error_messages:
            raise InvalidConfigError("\n".join(error_messages))

    def _get_training_context(self):
        """Capture training state for error diagnostics"""
        return {
            'current_strategy': self.active_strategy,
            'recent_rewards': list(self.performance_history)[-10:],
            'memory_usage': len(self.memory),
            'gradient_norms': {
                agent_id: self._get_gradient_norm(agent)
                for agent_id, agent in self.agents.items()
            }
        }

    def _get_gradient_norm(self, agent):
        """Safety check for gradient stability"""
        if hasattr(agent, 'policy_net'):
            params = [p.grad for p in agent.policy_net.parameters() if p.grad is not None]
            return torch.norm(torch.stack([torch.norm(p) for p in params])).item()
        return 0.0

    def _execute_recovery_strategy(self):
        """Hierarchical recovery system"""
        strategy_level = min(self.consecutive_errors // 3, len(self.recovery_strategies)-1)
        return self.recovery_strategies[strategy_level]()

    def _recover_soft_reset(self):
        """Level 1 Recovery: Reset network weights and clear buffers"""
        self.logger.warning("Executing soft reset")
        for agent in self.agents.values():
            if hasattr(agent, 'reset_parameters'):
                agent.reset_parameters()
        self.memory.clear()
        return {'status': 'recovered', 'strategy': 'soft_reset'}

    def _recover_learning_rate_adjustment(self):
        """Level 2 Recovery: Adaptive learning rate scaling"""
        self.logger.warning("Adjusting learning rates")
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'learning_rate'):
                agent.learning_rate *= 0.5
                agent.learning_rate = max(agent.learning_rate, 1e-6)
        return {'status': 'recovered', 'strategy': 'lr_adjustment'}

    def _recover_strategy_switch(self):
        """Level 3 Recovery: Fallback to safe strategy"""
        self.logger.warning("Switching to safe strategy")
        self.active_strategy = 'rl'  # Default to basic RL
        if self.safety_agent:
            return self.safety_agent.execute({'task': 'emergency_override'})
        return {'status': 'recovered', 'strategy': 'strategy_switch'}

    def _recover_full_reset(self):
        """Level 4 Recovery: Complete system reset"""
        self.logger.critical("Performing full reset!")
        self.__init__(...)  # Reinitialize with original params
        return {'status': 'recovered', 'strategy': 'full_reset'}

        # Initialize sub-agents
    def _initialize_agents(self, env, performance_metrics, shared_memory, 
                 mutation_rate=0.1, top_k=2):
        self.env = env
        self.performance = performance_metrics
        self.shared_memory = shared_memory
        self.mutation_rate = mutation_rate
        self.top_k = top_k
        self.param_bounds = {
            'dqn': {
                'hidden_size': (64, 512),
                'learning_rate': (1e-5, 0.1),
                'batch_size': (32, 1024),
                'target_update_frequency': (50, 500)
            },
            'maml': {
                'meta_lr': (1e-5, 0.01),
                'inner_lr': (1e-4, 0.1),
                'adaptation_steps': (1, 10)
            },
            'rsi': {
                'memory_size': (1000, 50000),
                'exploration_rate': (0.01, 0.5),
                'plasticity': (0.1, 2.0)
            },
            'rl': {
                'learning_rate': (1e-4, 0.1),
                'discount_factor': (0.8, 0.999),
                'epsilon': (0.01, 1.0),
                'epsilon_decay': (0.9, 0.9999)
            }
        }

        # Learning state
        self.meta_update_interval = 100
        self.curiosity_beta = 0.2  # Pathak et al. (2017)
        self.ewc_lambda = 0.4  # Kirkpatrick et al. (2017)

    def _create_agent(self, agent_id, params):
        """Instantiate new agent with mutated parameters"""
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        if agent_id == 'dqn':
            return DQNAgent(
                state_dim=state_size,
                action_dim=action_size,
                config=params
            )
        elif agent_id == 'maml':
            return MAMLAgent(
                state_size=state_size,
                action_size=action_size,
                shared_memory=self.shared_memory,
                **params
            )
        elif agent_id == 'rsi':
            return RSI_Agent(
                state_size=state_size,
                action_size=action_size,
                shared_memory=self.shared_memory,
                config=params
            )
        elif agent_id == 'rl':
            return RLAgent(
                possible_actions=list(range(action_size)),
                learning_rate=params.get('learning_rate', 0.001),
                discount_factor=params.get('discount_factor', 0.99),
                epsilon=params.get('epsilon', 1.0)
            )
        raise ValueError(f"Unknown agent type: {agent_id}")

    def _select_action(self, state, agent_type):
        """ Safe action selection """
        action = super()._select_action(state, agent_type)

        # Validate with safety agent
        validation = self.safety_agent.validate_action({
            'state': state,
            'proposed_action': action,
            'agent_type': agent_type
        })

        if not validation['approved']:
            self.logger.warning(f"Unsafe action {action} detected, using corrected action")
            return validation['corrected_action']
            
        return action

    def _process_state(self, raw_state):
        """
        Use the SLAILM to process environment state into a semantic feature vector.
        """
        if not isinstance(raw_state, str):
            raise ValueError("Expected raw_state to be a string.")
    
        analysis = self.slai_lm.process_input(prompt="Agent observation", text=raw_state)
    
        # Basic check on result structure
        if not isinstance(analysis, dict) or "feature_vector" not in analysis:
            raise NaNException("SLAILM returned malformed or incomplete feature vector")
    
        features = analysis["feature_vector"]
    
        if isinstance(features, list):
            # Convert to tensor
            features = torch.tensor(features, dtype=torch.float32)
    
        # Safety: check for NaNs or explosion
        if torch.isnan(features).any():
            raise NaNException("Feature vector contains NaN values.")
    
        grad_norm = features.norm().item()
        if grad_norm > 1e3:  # Customizable explosion threshold
            raise GradientExplosionError(grad_norm)
    
        return features

    def _training_error_handler(func):
        """Decorator for error recovery in training methods"""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (NaNException, GradientExplosionError, InvalidActionError) as e:
                self.logger.error(f"Training error: {str(e)}")
                self.error_history.append({
                    'timestamp': datetime.now(),
                    'error_type': type(e).__name__,
                    'context': self._get_training_context()
                })
                self.consecutive_errors += 1
                return self._execute_recovery_strategy()
            finally:
                if self.consecutive_errors > 0:
                    self.consecutive_errors -= 1
        return wrapper

    @_training_error_handler
    def train_episode(self, agent_type="dqn"):
        """
        Execute one training episode with selected strategy
        
        Implements Hybrid Reward Architecture from:
        van Seijen et al. (2017). Hybrid Reward Architecture for RL
        """
        state = self._process_state(self.env.reset())
        action = self._select_action(state, agent_type)
        if not self.safety_agent.validate_action({'action': action})['approved']:
            action = self.safety_agent.apply_corrections()
        self.shared_memory.append('learning_experiences', episode_data)
        external_data = self.shared_memory.get('external_experiences', [])
        self.memory.extend(external_data)
        state = self.env.reset()
        total_reward = 0
        episode_data = []
        agent = self.agents.get(agent_type)
        if not agent:
            raise ValueError(f"No agent found for type '{agent_type}'")
        
        dqn_loss = agent.train()
        grad_norm = self._get_gradient_norm(agent)
        
        while True:
            action = self._select_action(state, agent_type)
            next_state, reward, done, _ = self.env.step(action)
            raw_next_state, reward, done, _ = self.env.step(action)
            next_state = self._process_state(raw_next_state)

            
            # Store experience
            episode_data.append((state, action, reward, next_state, done))
            self._process_feedback(state, action, reward)
            
            total_reward += reward
            state = next_state
            if done: break
        
        # Train on episode data
        loss = self._train_strategy(episode_data, agent_type)
        self.performance_history.append(total_reward)
        self.performance_metrics['dqn_loss'].append(dqn_loss)
        self.performance_metrics['maml_grad_norm'].append(grad_norm)
        self.shared_memory.set('learning_metrics', self.performance_metrics)
        
        # Update strategy weights
        self._update_strategy_weights(loss)
        
        return total_reward

    def _process_feedback(self, state, action, reward):
        """
        Process feedback for online learning, memory updates, or logging.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        agent_type = self.rl_algorithm or "rl"
        agent = self.agents.get(agent_type)

        if not agent:
            logger.warning(f"No agent found for feedback processing: {agent_type}")
            return

        if hasattr(agent, 'process_feedback'):
            # Custom feedback handler (if implemented by agent)
            agent.process_feedback(state, action, reward)
        elif hasattr(agent, 'remember'):
            # Generic memory-based learning
            agent.remember((state, action, reward))
        else:
            logger.debug(f"Agent '{agent_type}' does not support feedback processing.")

    @_training_error_handler
    def meta_learn(self, num_tasks=5):
        """
        Meta-learning phase using MAML
        
        Implements:
        Wang et al. (2020). Automating RL with Meta-Learning
        """
        logger.info("Starting meta-learning phase...")
        
        for _ in range(num_tasks):
            adapted_agent = self.agents["maml"].adapt(self._create_task_variation())
            reward = self._evaluate_adapted(adapted_agent)
            self._update_meta_params(reward)

    def _setup_continual_learning(self):
        """Advanced neuroplasticity-preserving continual learning system"""
        # Core EWC parameters
        self.ewc_lambda = 0.4  # Elastic Weight Consolidation strength
        self.synaptic_plasticity = 0.8  # Maintain capacity for new learning
        self.meta_update_interval = 100
        # Biological plasticity mechanisms
        self.neurogenesis_threshold = 0.65  # Threshold for adding new units
        self.dendritic_scaling = 1.2  # Factor for branch-specific plasticity
        self.glial_support_factor = 0.3  # Simulated glial cell support
        # Dual plasticity-consolidation system
        self.plastic_weights = defaultdict(float)  # Transient knowledge
        self.consolidated_weights = defaultdict(float)  # Long-term storage
        # Neuromodulatory components
        self.norepinephrine_level = 0.5  # Attention/alertness
        self.dopamine_decay_rate = 0.95  # Reward prediction decay
       
        self.synaptic_strengths = { # Structural plasticity parameters
            'excitatory': 1.4,
            'inhibitory': 0.6,
            'modulatory': 0.9
        }

        self.fisher_matrix = self._initialize_fisher_matrix( # Advanced Fisher Information Matrix for EWC++ 
            decay_factor=0.9,
            sparse_rep=True,
            moving_avg_window=10
        )

        # Spike-Timing Dependent Plasticity (STDP) simulation
        self.stdp_window = deque(maxlen=100)  # Temporal integration window
        self.stdp_parameters = {
            'tau_plus': 20.0,   # LTP time constant
            'tau_minus': 20.0,  # LTD time constant
            'A_plus': 0.1,      # LTP rate
            'A_minus': 0.12     # LTD rate
        }

        # Molecular mechanisms
        self.protein_synthesis = {
            'mTOR_activation': 0.7,          # Memory consolidation pathway
            'BDNF_level': 0.85,              # Neurotrophic support
            'amyloid_clearance_rate': 0.92, # Simulated waste removal
            'prion_like_propagation': 0.05  # Memory engram spread
        }

        # Neurovascular coupling simulation
        self.metabolic_support = {
            'glucose_uptake': 0.88,
            'oxygen_supply': 0.95,
            'waste_removal': 0.78
        }

    def _initialize_fisher_matrix(self, decay_factor, sparse_rep, moving_avg_window):
        """Hierarchical Fisher information tracking with decay and sparsity"""
        return {
            'global_fisher': defaultdict(float),
            'modular_fisher': defaultdict(lambda: defaultdict(float)),
            'decay_factor': decay_factor,
            'sparsity_threshold': 0.01 if sparse_rep else 0,
            'moving_average': deque(maxlen=moving_avg_window),
            'historical_variance': defaultdict(float)
        }

    def _update_neuroplasticity(self, episode_data):
        """Dynamic neuroplasticity regulation based on learning progress"""
        # Calculate plasticity balance
        recent_grad_norm = np.mean(list(self.performance_metrics['gradient_norms'][-10:]))
        consolidation_pressure = min(1.0, recent_grad_norm * self.ewc_lambda)
        
        # Adaptive metaplasticity (Bienenstock-Cooper-Munro rule)
        self.synaptic_plasticity = np.tanh(
            self.dopamine_level * 
            self.norepinephrine_level *
            consolidation_pressure
        )
        
        # Structural reorganization
        if self._requires_neurogenesis(episode_data):
            self._add_neuronal_units(
                growth_factor=0.1,
                connectivity_density=0.7
            )
        
        # Update molecular pathways
        self._regulate_protein_synthesis(
            learning_intensity=np.mean(episode_data['rewards']),
            memory_age=time.time() - self.last_consolidation
        )

    def _apply_biologically_constrained_learning(self, gradients):
        """Apply neuromodulated, structure-aware learning rules"""
        # Dendritic compartmentalization
        gradients = self._scale_by_dendritic_branches(gradients)
        
        # Glial-guided learning
        gradients = self._apply_glial_modulation(gradients)
        
        # STDP-based weight update
        return self._stdp_weight_update(gradients)

    def _scale_by_dendritic_branches(self, gradients):
        """Simulate dendritic tree computation through gradient modulation"""
        for param, grad in gradients.items():
            if 'hidden' in param.name:
                gradients[param] *= self.dendritic_scaling * self.glial_support_factor
        return gradients

    def _requires_neurogenesis(self, episode_data):
        """Determine if new neural units should be added"""
        novelty_score = self._calculate_novelty(episode_data)
        resource_availability = self.metabolic_support['glucose_uptake']
        return (novelty_score > self.neurogenesis_threshold and 
                resource_availability > 0.6)

    def _regulate_protein_synthesis(self, learning_intensity, memory_age):
        """Simulate molecular mechanisms of memory consolidation"""
        # mTOR activation based on learning intensity
        self.protein_synthesis['mTOR_activation'] = np.clip(
            0.2 + 0.8 * learning_intensity,
            0.1, 0.95
        )
       
        self.protein_synthesis['BDNF_level'] = 0.5 + 0.5 * np.exp(-memory_age/1e5) # BDNF regulation
        
        # Amyloid clearance based on metabolic support
        self.protein_synthesis['amyloid_clearance_rate'] = (
            self.metabolic_support['waste_removal'] * 
            self.glial_support_factor
        )

    def continual_learning_loop(self, total_episodes=5000):
        """
        Main lifelong learning loop implementing:
        Ring (1997). CHILD: Continual Learning Framework
        """
        for ep in range(total_episodes):
            strategy = self._select_strategy()
            reward = self.train_episode(strategy)
            
            # Meta-learning updates
            if ep % self.meta_update_interval == 0:
                self.meta_learn(num_tasks=3)
                
            # RSI self-improvement
            if ep % self.rsi_improvement_cycle == 0:
                self._run_rsi_self_improvement()
                self._adapt_exploration_strategy()

            # Main training with meta-informed strategy selection
            strategy = self.strategy_evaluator.select_strategy()
            reward = self.train_episode(strategy)
            
    def _setup_trigger_system(self):
        """Initialize trigger detection parameters"""
        self.performance_threshold = 0.7  # Relative performance
        self.data_change_threshold = 0.15  # KL-divergence threshold
        self.retraining_interval = timedelta(hours=24)

    def run_learning_cycle(self):
        """Main learning orchestration loop"""
        if self.concept_drift_detector.analyze():
            self._adjust_learning_strategies()
            self._replay_historical_data()

    def _adjust_learning_strategies(self):
        """Dynamic strategy reweighting based on concept drift detection"""
        # Calculate recent performance variance
        recent_perf = list(self.performance_history)[-100:]
        if len(recent_perf) < 10:
            return

        # Calculate strategy effectiveness metrics
        perf_mean = np.mean(recent_perf)
        perf_std = np.std(recent_perf)
        volatility = perf_std / (abs(perf_mean) + 1e-9)

        # Adaptive strategy weighting
        if volatility > 0.5:  # High uncertainty regime
            # Favor exploration-heavy strategies
            self.strategy_weights = np.array([0.1, 0.3, 0.1, 0.5])  # Boost RSI
        else:  # Stable regime
            # Favor exploitation-optimized strategies
            self.strategy_weights = np.array([0.2, 0.4, 0.3, 0.1])  # Favor DQN/MAML

        # Apply momentum to smooth transitions
        self.strategy_weights = 0.7 * self.strategy_weights + 0.3 * np.ones(4)/4

        # Parameter adjustments
        for agent_id, agent in self.agents.items():
            # Increase exploration in volatile phases
            if hasattr(agent, 'epsilon'):
                new_epsilon = min(1.0, agent.epsilon * (1.2 if volatility > 0.5 else 0.95))
                agent.epsilon = max(new_epsilon, 0.01)

            # Adjust learning rates inversely with performance stability
            if hasattr(agent, 'learning_rate'):
                lr_factor = 1.0 + (perf_std / (abs(perf_mean) + 1e-9))
                agent.learning_rate = np.clip(
                    agent.learning_rate * lr_factor,
                    1e-5, 0.1
                )

    def _replay_historical_data(self):
        """Experience replay with prioritized historical sampling"""
        # Retrieve historical data from shared memory
        historical_data = self.shared_memory.get('historical_episodes', [])
        if not historical_data:
            return

        # Hybrid replay sampling
        replay_strategy = 'prioritized' if len(historical_data) > 100 else 'uniform'
        
        if replay_strategy == 'prioritized':
            # Simple temporal prioritization (recent experiences first)
            replay_data = sorted(historical_data, 
                            key=lambda x: x['timestamp'], 
                            reverse=True)[:100]
        else:
            replay_data = random.sample(historical_data, 
                                    min(len(historical_data), 100))

        # Batch replay training
        for episode in replay_data:
            # Convert stored data to training format
            states = episode.get('states', [])
            actions = episode.get('actions', [])
            rewards = episode.get('rewards', [])
            
            if len(states) < 2:
                continue

            # Train each agent with historical data
            for agent_id, agent in self.agents.items():
                if agent_id == 'dqn' and hasattr(agent, 'store_transition'):
                    # Convert to DQN's transition format
                    for i in range(len(states)-1):
                        agent.store_transition(
                            states[i], actions[i], rewards[i], 
                            states[i+1], False
                        )
                    if len(agent.memory) > agent.batch_size:
                        agent.train()
                        
                elif agent_id == 'rl' and hasattr(agent, 'learn'):
                    # Update Q-values directly from historical traces
                    for i in range(len(states)-1):
                        current_state = tuple(states[i])
                        next_state = tuple(states[i+1])
                        agent.learn(next_state, rewards[i], False)

                # Additional agent-specific replay logic can be added here

        # Clear old memories to prevent overfitting
        if len(historical_data) > 1000:
            self.shared_memory.set('historical_episodes', historical_data[-1000:])

    def _check_learning_triggers(self):
        """Evaluate activation conditions for learning"""
        return any([
            self._detect_new_data(),
            self._check_performance_drop(),
            self._detect_concept_drift(),
            self._check_scheduled_retraining()
        ])

    def _execute_continual_learning(self):
        """Execute comprehensive learning pipeline"""
        # Phase 1: Strategy Evaluation
        strategy_performance = self._evaluate_strategies()
        
        # Phase 2: Catastrophic Forgetting Prevention
        self._apply_ewc_regularization()
        
        # Phase 3: Meta-Learning Update
        if self._requires_meta_update():
            self._run_meta_learning_phase()
        
        # Phase 4: Evolutionary Optimization
        optimized_agents = self._evolve_strategies(strategy_performance)
        self._update_agent_pool(optimized_agents)

    def _select_strategy(self):
        """Thompson sampling for strategy selection"""
        sampled = np.random.normal(self.strategy_weights, 0.1)
        return ["rl", "dqn", "maml", "rsi"][np.argmax(sampled)]

    def _train_strategy(self, episode_data, agent_type):
        """Agent-specific training logic"""
        agent = self.agents[agent_type]
        
        if agent_type == "rl":
            return agent.execute(episode_data)["loss"]
        elif agent_type == "dqn":
            for transition in episode_data:
                agent.store_transition(*transition)
            return agent.train()
        elif agent_type == "maml":
            return agent.meta_update([(self.env, episode_data)])
        elif agent_type == "rsi":
            agent.remember(episode_data)
            return agent.train()
        
        return 0.0

    def _evaluate_strategies(self):
        """Assess all learning strategies using validation tasks"""
        return {agent_id: agent.evaluate() 
                for agent_id, agent in self.agents.items()}

    def _apply_ewc_regularization(self):
        """Implement Elastic Weight Consolidation"""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'consolidate_weights'):
                agent.consolidate_weights(self.ewc_lambda)

    def _evolve_strategies(self, performance_data):
        """Evolutionary strategy optimization"""
        evolutionary_factory = EvolutionaryFactory(
            env=self.env,
            performance_metrics=performance_data,
            mutation_rate=0.15
        )
        return evolutionary_factory.generate_new_strategies()

    def _run_rsi_self_improvement(self):
        """RSI optimization cycle"""
        analysis = self.agents["rsi"].execute({
            "performance_history": self.performance_history,
            "strategy_weights": self.strategy_weights
        })
        logger.info(f"RSI optimization result: {analysis}")

    def _update_strategy_weights(self, rewards):
        strategy_performance = {
            agent_id: np.mean(rewards[agent_id][-10:]) 
            for agent_id in self.agents
        }
        total = sum(strategy_performance.values())
        self.strategy_weights = np.array([strategy_performance[id]/total for id in self.agents.keys()])

    def _detect_new_data(self):
        """Check shared memory for new data flags"""
        return self.shared_memory.get('new_data_flag', False)

    def _check_performance_drop(self):
        """Calculate relative performance degradation"""
        reward_data = list(self.performance_metrics['reward'])
        if not reward_data:
            return False
            
        recent_perf = np.mean(reward_data[-100:])
        max_perf = np.max(reward_data)
        return recent_perf < (self.performance_threshold * max_perf)

    def _detect_concept_drift(self):
        """Statistical test for distribution shifts"""
        return self.concept_drift_detector.analyze(
            self.shared_memory.get('data_stream')
        )

    def _check_scheduled_retraining(self):
        """Time-based retraining trigger"""
        return datetime.now() > self.last_retraining + self.retraining_interval

    # Core Learning Components
    def _run_meta_learning_phase(self):
        """Execute MAML-based meta-learning"""
        task_pool = self._generate_meta_tasks()
        meta_loss = self.agents['maml'].meta_update(task_pool)
        self.performance_metrics['meta_loss'].append(meta_loss)

    def _generate_meta_tasks(self):
        """Create task variations for meta-learning"""
        return [self._create_task_variation() for _ in range(3)]

    def _create_task_variation(self):
        import copy
        """Create new task variant through controlled parameter randomization
        without requiring environment-specific knowledge"""
        try:
            # Create a deep copy to avoid modifying original environment
            env_variant = copy.deepcopy(self.env)
            
            # Generic parameter randomization
            if hasattr(env_variant.unwrapped, 'gravity'):
                # Physics-based environment modification
                env_variant.unwrapped.gravity = np.clip(
                    self.env.unwrapped.gravity * np.random.uniform(0.8, 1.2),
                    0.5 * self.env.unwrapped.gravity,
                    2.0 * self.env.unwrapped.gravity
                )
                
            if hasattr(env_variant.unwrapped, 'mass'):
                # Mass property randomization
                env_variant.unwrapped.mass = np.clip(
                    self.env.unwrapped.mass * np.random.uniform(0.5, 2.0),
                    0.1 * self.env.unwrapped.mass,
                    5.0 * self.env.unwrapped.mass
                )
                
            if hasattr(env_variant.unwrapped, 'tau'):
                # Control timing parameter variation
                env_variant.unwrapped.tau = np.clip(
                    self.env.unwrapped.tau * np.random.uniform(0.9, 1.1),
                    0.001, 0.05
                )
                
            # Generic reward shaping variation
            reward_weights = {
                'time': np.random.uniform(0.8, 1.2),
                'survival': np.random.uniform(0.5, 1.5),
                'action_cost': np.random.uniform(0.7, 1.3)
            }
            env_variant.unwrapped.reward_weights = reward_weights
            
            # Stochastic dynamics injection
            if hasattr(env_variant, 'model'):
                # For Mujoco-based environments
                for joint in env_variant.model.jnt_names:
                    idx = env_variant.model.joint_name2id(joint)
                    env_variant.model.dof_damping[idx] *= np.random.uniform(0.8, 1.2)
                    
            # Add observation noise if no physical parameters changed
            if env_variant == self.env:
                env_variant.observation_space = self._add_observation_noise()
                
            params = {
                'gravity': getattr(env_variant.unwrapped, 'gravity', None),
                'mass': getattr(env_variant.unwrapped, 'mass', None),
                'reward_weights': reward_weights
            }
            self.logger.info(f"Created task variant with params: {params}")
            
            return env_variant
            
        except Exception as e:
            self.logger.warning(f"Task variation failed: {str(e)}")
            # Fallback: Add observation noise to original environment
            return self._add_observation_noise()

    def _add_observation_noise(self):
        """Create observation space variation through noise injection"""
        class NoisyEnvWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.noise_level = np.random.uniform(0.01, 0.1)
                
            def reset(self, **kwargs):
                state = self.env.reset(**kwargs)
                return state + self.noise_level * np.random.randn(*state.shape)
                
            def step(self, action):
                state, reward, done, truncated, info = self.env.step(action)
                noisy_state = state + self.noise_level * np.random.randn(*state.shape)
                return noisy_state, reward, done, truncated, info
                
        return NoisyEnvWrapper(self.env)

    def _evaluate_adapted(self, agent):
        """Evaluate adapted policy on modified task"""
        total_reward = 0
        state = self.env.reset()
        
        while True:
            action = agent.act(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done: break
            
        return total_reward

    def _update_meta_params(self, reward):
        """Update meta-parameters based on adaptation success"""
        self.strategy_weights[2] *= (1 + reward/1000)  # MAML weight
        self.curiosity_beta = max(0.1, self.curiosity_beta * (1 + reward/500))

    def _update_shared_knowledge(self):
        """Sync learned parameters with shared memory"""
        knowledge_package = {
            'policy_weights': self._extract_policy_parameters(),
            'performance_stats': dict(self.performance_metrics),
            'strategy_distribution': self.strategy_weights
        }
        self.shared_memory.update('learning_state', knowledge_package)

    def _extract_policy_parameters(self):
        """Gather policy parameters from all agents"""
        return {agent_id: agent.get_parameters()
                for agent_id, agent in self.agents.items()
                if hasattr(agent, 'get_parameters')}

class MetaStrategyEvaluator:
    """Decides strategy weighting using meta-learning insights"""
    def __init__(self, agents, performance_metrics):
        self.agents = agents
        self.metric_history = performance_metrics
        self.strategy_candidates = ['rl', 'dqn', 'maml', 'rsi']
        
    def select_strategy(self):
        """Context-aware strategy selection"""
        recent_perf = {s: np.mean(self.metric_history[s][-10:]) 
                      for s in self.strategy_candidates}
        
        # Meta-learning based weighting
        weights = np.array([
            recent_perf['rl']**2,  # Basic RL stability
            recent_perf['dqn'] * 1.5,  # DQN sample efficiency
            self.agents['maml'].meta_loss * -0.1 if hasattr(self.agents['maml'], 'meta_loss') else 1.0,
            recent_perf['rsi'] * 0.8  # RSI long-term adaptation
        ])
        
        # Softmax normalization
        exp_weights = np.exp(weights - np.max(weights))
        return np.random.choice(
            self.strategy_candidates, 
            p=exp_weights/exp_weights.sum()
        )

class ConceptDriftDetector:
    """Statistical concept drift detection system using multivariate Gaussian KL-divergence"""
    
    def __init__(self, window_size=100, threshold=0.15, epsilon=1e-8):
        self.data_window = deque(maxlen=window_size)
        self.threshold = threshold
        self.epsilon = epsilon  # Numerical stability constant

    def analyze(self, data_stream):
        """Detect distribution shifts using KL-divergence between historical and current data"""
        if not data_stream or len(data_stream) < 10:  # Minimum data requirement
            return False
            
        # Convert to numpy arrays and check dimensions
        current_batch = np.array(list(data_stream)[-len(self.data_window):])
        historical_data = np.array(self.data_window)
        
        # Handle initial window population
        if len(historical_data) < 10:
            self.data_window.extend(current_batch)
            return False

        # Calculate KL divergence
        kl_div = self._calculate_kl_divergence(historical_data, current_batch)
        self.data_window.extend(current_batch)
        
        return kl_div > self.threshold

    def _calculate_kl_divergence(self, p, q):
        """Compute KL(P||Q) between two multivariate Gaussian distributions"""
        # Epsilon to prevent singular matrices
        p += np.random.normal(0, self.epsilon, p.shape)
        q += np.random.normal(0, self.epsilon, q.shape)
        
        # Calculate means and covariance matrices
        mu_p = np.mean(p, axis=0)
        sigma_p = np.cov(p, rowvar=False) + np.eye(p.shape[1])*self.epsilon
        mu_q = np.mean(q, axis=0)
        sigma_q = np.cov(q, rowvar=False) + np.eye(q.shape[1])*self.epsilon
        
        # KL divergence formula for multivariate Gaussians
        diff = mu_q - mu_p
        sigma_q_inv = np.linalg.inv(sigma_q)
        n = mu_p.shape[0]
        
        trace_term = np.trace(sigma_q_inv @ sigma_p)
        quad_form = diff.T @ sigma_q_inv @ diff
        logdet_term = np.log(np.linalg.det(sigma_q)/np.linalg.det(sigma_p))
        
        return 0.5 * (trace_term + quad_form - n + logdet_term)

class EvolutionaryFactory:
    """Evolutionary strategy optimization factory with parameter mutation"""
    
    def __init__(self, env, performance_metrics, shared_memory, 
                 mutation_rate=0.1, top_k=2):
        self.env = env
        self.performance = performance_metrics
        self.shared_memory = shared_memory
        self.mutation_rate = mutation_rate
        self.top_k = top_k
        self.param_bounds = {
            'dqn': {
                'hidden_size': (64, 512),
                'learning_rate': (1e-5, 0.1),
                'batch_size': (32, 1024)
            },
            'maml': {
                'meta_lr': (1e-5, 0.01),
                'inner_lr': (1e-4, 0.1),
                'adaptation_steps': (1, 10)
            },
            'rsi': {
                'memory_size': (1000, 50000),
                'exploration_rate': (0.01, 0.5),
                'plasticity': (0.1, 2.0)
            },
            'rl': {
                'learning_rate': (1e-4, 0.1),
                'discount_factor': (0.8, 0.999),
                'epsilon': (0.01, 1.0),
                'epsilon_decay': (0.9, 0.9999)
            }
        }

    def generate_new_strategies(self):
        """Generate new agents through selection, mutation, and crossover"""
        optimized_agents = []
        
        # Select top performing agents
        sorted_agents = sorted(self.performance.items(), 
                             key=lambda x: x[1], reverse=True)[:self.top_k]
        
        for agent_id, _ in sorted_agents:
            # Create mutated variants
            for _ in range(2):  # Generate 2 variants per top agent
                mutated_params = self._mutate_parameters(agent_id)
                optimized_agents.append(
                    self._create_agent(agent_id, mutated_params)
                )
        
        # Add crossover between top performers
        if len(sorted_agents) >= 2:
            hybrid_params = self._crossover(sorted_agents[0][0], sorted_agents[1][0])
            optimized_agents.append(
                self._create_agent(sorted_agents[0][0], hybrid_params)
            )
        
        return optimized_agents

    def _mutate_parameters(self, agent_id):
        """Apply Gaussian mutation to parameters within defined bounds"""
        params = {}
        for param, (min_val, max_val) in self.param_bounds[agent_id].items():
            # Get base value from current best parameters
            base_val = (max_val + min_val)/2  # In real implementation, use actual current values
            # Apply mutation
            mutated = base_val * (1 + self.mutation_rate * np.random.randn())
            params[param] = np.clip(mutated, min_val, max_val)
            
            # Round integer parameters
            if param in ['hidden_size', 'batch_size', 'memory_size', 'adaptation_steps']:
                params[param] = int(params[param])
                
        return params

    def _create_agent():
        pass

    def _crossover(self, agent_id1, agent_id2):
        """Combine parameters from two different agent types"""
        common_params = set(self.param_bounds[agent_id1]) & set(self.param_bounds[agent_id2])
        hybrid_params = {}
        for param in common_params:
            if np.random.rand() > 0.5:
                hybrid_params[param] = self.param_bounds[agent_id1][param][1]
            else:
                hybrid_params[param] = self.param_bounds[agent_id2][param][1]
        return hybrid_params

__all__ = ['LearningAgent', 'SLAIEnv']
