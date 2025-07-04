__version__ = "1.9.0"

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
import copy
import time
import math
import psutil
import random
import functools
import numpy as np
import torch.nn as nn
import gymnasium as gym

from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, Union, Tuple, Optional, Any
from functools import partial

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.learning.utils.error_calls import (NaNException, GradientExplosionError,
                                                   InvalidActionError, InvalidConfigError)
from src.agents.learning.utils.multi_task_learner import MultiTaskLearner
from src.agents.learning.utils.state_processor import StateProcessor
from src.agents.learning.utils.recovery_system import RecoverySystem
from src.agents.learning.learning_calculations import LearningCalculations
from src.agents.learning.strategy_selector import StrategySelector
from src.agents.learning.learning_factory import LearningFactory
from src.agents.learning.dqn import DQNAgent
from src.agents.learning.maml_rl import MAMLAgent
from src.agents.learning.rsi import RSIAgent
from src.agents.learning.rl_agent import RLAgent
from src.agents.learning.slaienv import SLAIEnv
from src.agents.base.light_metric_store import LightMetricStore
from src.agents.base.lazy_agent import LazyAgent
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Learning Agent")
printer = PrettyPrinter

def validation_logic(self):
    required_params = {
        'dqn': ['hidden_size', 'gamma'],
        'maml': ['meta_lr', 'inner_lr']
    }
    for agent_type, params in required_params.items():
        if not all(k in self.config[agent_type] for k in params):
            raise ValueError(f"Missing params for {agent_type}: {params}")

class LearningAgent(BaseAgent):
    """Orchestrates SLAI's lifelong learning capabilities through multiple strategies"""

    def __init__(self,
                 shared_memory,
                 agent_factory,
                 env=SLAIEnv, config=None,
                 args=(), kwargs={}):
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
        self.env = env
        self.task_ids = ['default_task'] # Example task list – adapt based on actual environment capabilities ['navigate', 'explore', 'avoid', 'collect'] 
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.learning_config = get_config_section('learning_agent')

        state_dim = env.state_dim if isinstance(env, SLAIEnv) else self.learning_config.get('state_dim')
        self.rl_algorithm = self.config.get("rl_algorithm", None)

        self.batch_size = self.learning_config.get('batch_size', 32)
        self.strategy_weights = np.array(self.learning_config.get('strategy_weights', [0.25]*4))
        self.prediction_weights = self.learning_config.get('prediction_weights', [0.25]*4)
        self.maml_task_pool_size = self.learning_config.get('maml_task_pool_size', 100)
        self.rsi_improvement_cycle = self.learning_config.get('rsi_improvement_cycle', 50)
        self.performance_threshold = self.learning_config.get('performance_threshold', 0.7)
        self.data_change_threshold = self.learning_config.get('data_change_threshold', 0.15)
        self.retraining_interval = timedelta(hours=self.learning_config.get('retraining_interval_hours', 24))
        self.novelty_threshold = self.learning_config.get('novelty_threshold', 0.3)
        self.uncertainty_threshold = self.learning_config.get('uncertainty_threshold', 0.25)
        self.maml_adaptation_steps = self.learning_config.get('maml_adaptation_steps', 10)
        
        # Initialize buffers with config sizes
        self.embedding_buffer = deque(maxlen=self.learning_config.get('embedding_buffer_size', 512))
        self.performance_history = deque(maxlen=self.learning_config.get('performance_history_size', 1000))
        self.state_recency = deque(maxlen=self.learning_config.get('state_recency_size', 1000))
        self.architecture_history = deque(maxlen=self.learning_config.get('architecture_history_size', 10))
        self.task_embedding_dim = self.learning_config.get("task_embedding_dim", 256)

        # Determine state_dim based on the env passed at initialization
        if isinstance(env, SLAIEnv) and hasattr(env, 'state_dim'):
            self.state_dim = env.state_dim
        elif hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
             self.state_dim = env.observation_space.shape[0]
        else:
            # Fallback to config if env doesn't provide it directly
            self.state_dim = self.learning_config.get('state_dim', 10) # Default from config
        
        # action_dim determination
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
            self.action_dim = env.action_space.n
        else:
            self.action_dim = self.learning_config.get('action_dim', 2) # Default from config

        # State embedding layer
        self.state_embedder = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.task_embedding_dim)
        )
        
        # Deferred initialization config
        deferred_config = self.learning_config.get('deferred_init', {})
        self._config = {
            'max_network_size': deferred_config.get('max_network_size', 256),
            'max_task_pool': deferred_config.get('max_task_pool', 50),
            'max_history': deferred_config.get('max_history', 500)
        }

        # Meta-controller (self.policy_net) configuration
        meta_controller_config = self.learning_config.get('meta_controller', {})
        self.task_embedding_dim = meta_controller_config.get('task_embedding_dim', 256) # Dimension of task/state embeddings
        
        self.agent_strategies_map = {
            'dqn': 0,
            'maml': 1,
            'rsi': 2,
            'rl': 3,
            'planning': 4
        }
        self.num_agent_strategies = len(self.agent_strategies_map)

        # policy_net in LearningAgent is for meta-control: predicting best agent strategy
        self.policy_net = nn.Sequential(
            nn.Linear(self.task_embedding_dim, meta_controller_config.get('hidden_dim', 128)),
            nn.ReLU(),
            nn.Linear(meta_controller_config.get('hidden_dim', 128), self.num_agent_strategies) # Outputs logits for each strategy
        )
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.state_embedder.parameters()),
            lr=meta_controller_config.get('learning_rate', 1e-3))
        self.loss_fn = nn.CrossEntropyLoss()
        self.training_mode = False
        self.device = 'cpu'

        self.performance_metrics = {}
        self.recovery_system = RecoverySystem(learning_agent=self)
        self.strategy_selector = StrategySelector(
            config=self.learning_config,
            agent_strategies_map=self.agent_strategies_map,
            state_embedder=self.state_embedder,
            policy_net=self.policy_net,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device
        )
        self.learning_calculations = LearningCalculations()
        self.learning_factory = LearningFactory(
            env=env,
            performance_metrics=self.performance_metrics
        )
        self.state_embedder = self.strategy_selector.state_embedder
        self.performance_metrics = {
            # These are useful across all agent types:
            'scenario_rewards': defaultdict(float),        # Mean reward per scenario or task
            'success_rate': defaultdict(float),            # Percentage of episodes meeting success criteria
            'episode_length': defaultdict(list),           # Track duration of episodes
            'q_value_mean': defaultdict(list),             # Mean Q-value per evaluation run
            'q_value_std': defaultdict(list),              # Std of Q-values (to track learning stability)

            # Per agent strategy (DQN, MAML, RSI, RL):
            'strategy_selection_count': defaultdict(int),  # How often each strategy is selected
            'strategy_accuracy': defaultdict(float),       # Meta-controller prediction accuracy
            'strategy_loss': defaultdict(float),           # Loss during meta-controller training

            # For adaptive tuning and monitoring drift or stagnation:
            'novelty_score': defaultdict(float),           # From novelty detector (e.g., for RSI)
            'uncertainty_estimate': defaultdict(float),    # Optional: from model prediction variance
            'catastrophic_forgetting': defaultdict(float), # Score or heuristic for forgetting events
            'concept_drift_detected': defaultdict(bool),   # Boolean or counter per task/episode

            # For experience buffer and MAML/meta-training:
            'embedding_buffer_size': lambda: len(self.embedding_buffer),
            'replay_buffer_usage': defaultdict(int),       # Transitions used per training step
            'performance_history_stats': lambda: {
                'mean': np.mean(self.performance_history) if self.performance_history else 0,
                'std': np.std(self.performance_history) if self.performance_history else 0,
            },

            # Specific to LearningFactory and RSI:
            'param_mutation_rate': {},                     # Track over time for RSI
            'checkpoint_quality': defaultdict(float),      # Loaded from model checkpoints
            'agent_fitness_score': defaultdict(float),     # For evolutionary scoring of agents

            'plot_tags': ['average_reward', 'success_rate', 'strategy_selection_count', 'novelty_score'],
        }
        self.state_processor = StateProcessor(env)
        self.multi_task_learner = MultiTaskLearner(task_ids=self.task_ids)
        self.architecture_history = deque(maxlen=self.learning_config.get('architecture_history_size', 10))

        self.observation_count = 0

        logger.info(f"Learning Agent has succesfully initialized with meta-controller for {self.num_agent_strategies} strategies.")
        logger.info(f"Meta-controller input (task embedding) dimension: {self.task_embedding_dim}")
        print(f"Initialized with state_dim: {self.state_dim}")

    @property
    def performance_metrics(self):
        """Getter for performance metrics"""
        return self._performance_metrics
    
    @performance_metrics.setter
    def performance_metrics(self, value):
        """Setter for performance metrics with initialization logic"""
        self._performance_metrics = value or {}

        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss

        # Initialize learning subsystems
        self._initialize_agents(
            env=self.env,
            performance_metrics=self.performance_metrics
        )
        if self.env is None or not hasattr(self.env, 'observation_space') or not hasattr(self.env, 'action_space'):
            raise ValueError("LearningAgent requires a valid environment with observation_space and action_space.")
        else:
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n

        possible_actions = list(range(action_dim))

        state_size = state_dim
        action_size = action_dim
        self.agents = {
            'dqn': DQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    agent_id="dqn_agent",
                ),
            'maml': MAMLAgent(
                    state_size=state_size,
                    action_size=action_size,
                    agent_id="maml_agent"
                ),
            'rsi': RSIAgent(
                    state_size=state_size,
                    action_size=action_size,
                    agent_id="rsi_agent"
                ),
            'rl': RLAgent(
                    possible_actions=list(range(action_dim)),
                    state_size=state_size,
                    agent_id="rl_agent"
                )
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

        # Defer heavy initialization
        self._deferred_initialization()

        self._performance_metrics = LightMetricStore()
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

        # Configuration with memory limits
        self._config = {
            'max_network_size': 256,  # Hidden units
            'max_task_pool': 50,
            'max_history': 500
        }

    def _create_dqn_agent(self):
        """Create DQN agent with optimized network"""
        return DQNAgent(
            state_dim=self.state_dim,  # self.env.observation_space.shape[0],
            action_dim=self.action_dim,  # self.env.action_space.n,
            agent_id="dqn_agent"
        )

    def _create_maml_agent(self):
        """Create MAML agent with shared network components"""
        return MAMLAgent(
            state_size=self.state_dim,  # self.env.observation_space.shape[0],
            action_size=self.action_dim,  # self.env.action_space.n,
            agent_id="maml_agent"
        )

    def _create_rsi_agent(self):
        """Create RSI agent with memory limits"""
        return RSIAgent(
            state_size=self.state_dim,  # self.env.observation_space.shape[0],
            action_size=self.action_dim,  # self.env.action_space.n,
            agent_id="rsi_agent"
        )

    def _create_rl_agent(self):
        """Create basic RL agent"""
        return RLAgent(
            state_size=self.state_dim,  # self.env.observation_space.shape[0],
            possible_actions=list(range(self.action_dim)),  # list(range(self.env.action_space.n)),
            agent_id="rl_agent"
        )

    def select_agent_strategy_with_meta_controller(
        self,
        current_task_env_or_state_embedding,
        episode_results: dict
    ) -> str:
        """
        Robust strategy selection via meta-controller.
    
        Args:
            current_task_env_or_state_embedding (Union[np.ndarray, torch.Tensor]):
                Task-level embedding or state vector.
            episode_results (dict): Metrics from last episode (reward, success rate, etc.)
    
        Returns:
            str: Strategy name chosen by meta-controller (dqn, maml, rsi, rl)
        """
    
        # === Step 1: Validate Input Presence ===
        if current_task_env_or_state_embedding is None:
            self.logger.error("Input embedding is None.")
            return self.default_strategy_name
    
        # === Step 2: Convert Input to Torch Tensor ===
        try:
            if isinstance(current_task_env_or_state_embedding, torch.Tensor):
                embedding = current_task_env_or_state_embedding.float()
            elif isinstance(current_task_env_or_state_embedding, np.ndarray):
                if current_task_env_or_state_embedding.dtype == np.object_:
                    current_task_env_or_state_embedding = np.array(
                        current_task_env_or_state_embedding.tolist(), dtype=np.float32
                    )
                embedding = torch.tensor(current_task_env_or_state_embedding, dtype=torch.float32)
            elif isinstance(current_task_env_or_state_embedding, (list, tuple)):
                embedding = torch.tensor(current_task_env_or_state_embedding, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported embedding type: {type(current_task_env_or_state_embedding)}")
        except Exception as e:
            self.logger.error("Embedding conversion failed", exc_info=True)
            return self.default_strategy_name
    
        # === Step 3: Ensure Correct Shape ===
        try:
            if embedding.ndim == 1:
                embedding = embedding.unsqueeze(0)  # Shape (1, D)
            elif embedding.ndim != 2:
                raise ValueError(f"Embedding must be 1D or 2D, got shape {embedding.shape}")
        except Exception as e:
            self.logger.error("Embedding shape validation failed", exc_info=True)
            return self.default_strategy_name
    
        # === Step 4: Pad or Truncate ===
        expected_dim = self.task_embedding_dim
        actual_dim = embedding.shape[1]
        if actual_dim < expected_dim:
            padding = torch.zeros((embedding.shape[0], expected_dim - actual_dim), dtype=embedding.dtype)
            embedding = torch.cat([embedding, padding], dim=1)
        elif actual_dim > expected_dim:
            embedding = embedding[:, :expected_dim]
    
        # === Step 5: Check for Meta-Controller ===
        if not hasattr(self, "policy_net") or self.policy_net is None:
            self.logger.warning("Meta-controller (policy_net) not defined.")
            return self.default_strategy_name
    
        # === Step 6: Device Matching ===
        try:
            device = next(self.policy_net.parameters()).device
            embedding = embedding.to(device)
        except Exception as e:
            self.logger.error("Device mismatch while preparing embedding", exc_info=True)
            return self.default_strategy_name
    
        # === Step 7: Inference ===
        try:
            self.policy_net.eval()
            with torch.no_grad():
                logits = self.policy_net(embedding)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits.squeeze(0), dim=0)
                strategy_index = torch.argmax(probs).item()
        except Exception as e:
            self.logger.error("Meta-controller inference failed", exc_info=True)
            return self.default_strategy_name
    
        # === Step 8: Resolve Strategy ===
        strategy_names = list(self.agents.keys())
        if not strategy_names:
            self.logger.critical("No agent strategies available.")
            return self.default_strategy_name
    
        if strategy_index >= len(strategy_names):
            self.logger.warning(f"Strategy index {strategy_index} out of bounds. Falling back.")
            strategy_index = len(strategy_names) - 1
    
        selected_strategy = strategy_names[strategy_index]
        if selected_strategy not in self.agents:
            self.logger.error(f"Strategy '{selected_strategy}' not found in agent registry.")
            return self.default_strategy_name
    
        self.logger.info(f"Meta-controller selected strategy: {selected_strategy}")
        return selected_strategy

    def observe(self, task_embedding, best_agent_strategy_name):
        self.strategy_selector.observe(task_embedding, best_agent_strategy_name)

    def train_from_embeddings(self):
        return self.strategy_selector.train_from_embeddings()

    def select_agent_strategy(self, state_embedding):
        return self.strategy_selector.select_strategy(state_embedding)

    def _generate_task_embedding_and_label(self, task_env, episode_results: Dict[str, float]):
        """
        Generate task embedding and best strategy label from environment and results.
        Uses multi-source feature extraction.
        """
        # Validate inputs
        if not episode_results or len(episode_results) == 0:
            logger.warning("No episode results for embedding generation")
            return np.zeros(self.task_embedding_dim), "unknown"
            
        # Feature extraction
        features = []
        
        # 1. Environment structural features
        if hasattr(task_env, 'observation_space'):
            obs_shape = task_env.observation_space.shape
            features.append(obs_shape[0] if obs_shape else 0)
            
        if hasattr(task_env, 'action_space'):
            features.append(task_env.action_space.n)
            
        # 2. Performance characteristics
        perf_features = [
            np.mean(list(dict(episode_results).values())),  # Avg performance
            np.std(list(episode_results.values())),         # Performance variance
            len(episode_results)                            # Number of strategies
        ]
        features.extend(perf_features)
        
        # 3. Environment-specific parameters (if available)
        for param in ['gravity', 'friction_coeff', 'wind_strength']:
            if hasattr(task_env, param):
                features.append(getattr(task_env, param))
                
        # 4. Initial state characteristics
        try:
            initial_state, _ = task_env.reset()
            processed_state = self.state_processor.process(initial_state)
            features.extend(processed_state[:5])  # Use first 5 elements
        except Exception as e:
            logger.warning(f"State processing failed: {str(e)}")
            
        # Padding/truncation
        features = features[:self.task_embedding_dim]
        if len(features) < self.task_embedding_dim:
            features += [0.0] * (self.task_embedding_dim - len(features))
            
        # Convert to tensor
        task_embedding = torch.tensor(features, dtype=torch.float32)
        
        # Determine best strategy
        best_strategy = max(episode_results, key=episode_results.get)
        
        logger.debug(f"Generated embedding | Dim: {len(features)} | "
                     f"Best: {best_strategy} | Score: {episode_results[best_strategy]:.2f}")
        
        return task_embedding, best_strategy

    # Remove update_from_embeddings since it's redundant
    def update_from_embeddings(self, inputs, targets):
        logger.warning("Deprecated method called. Use train_from_embeddings instead.")
        return 0.0

    def _create_task_variation(self):
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

    def calculate_planning_reward(self, risks: int, opportunities: int) -> float:
        """Custom reward function for planning simulations"""
        risk_penalty = -0.5 * risks
        opportunity_bonus = 1.0 * opportunities
        novelty_bonus = self._calculate_novelty_bonus(self._current_state)
        return risk_penalty + opportunity_bonus + novelty_bonus

    def _calculate_reward(self, state, action, next_state, base_reward):
        """
        Enhanced reward calculation with multi-faceted components:
        1. Base environment reward
        2. Novelty bonus for unexplored states
        3. Consistency bonus for predictable behavior
        4. Energy cost penalty for computational expense
        5. Curiosity-driven intrinsic motivation
        """
        # 1. Base environment reward
        total_reward = base_reward
        
        # 2. Novelty bonus
        novelty_bonus = self._calculate_novelty_bonus(next_state)
        total_reward = base_reward
        total_reward += novelty_bonus * self._curiosity_scaling_factor()
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


        # 3. Consistency bonus
        consistency_bonus = self._calculate_consistency(state, action)
        total_reward += consistency_bonus
        
        # 4. Energy cost penalty
        energy_cost = self._calculate_energy_cost(action)
        total_reward -= energy_cost
        
        # 5. Curiosity-driven intrinsic motivation
        if hasattr(self, 'curiosity_module'):
            intrinsic_reward = self.learning_factory.compute_intrinsic_reward(state, action, next_state)
            total_reward += intrinsic_reward
            
        return total_reward

    def _calculate_novelty_bonus(self, state: np.ndarray) -> float:
        """
        Calculate multi-factor novelty bonus for less explored states.
        
        Args:
            state: Current state vector
            
        Returns:
            Novelty bonus value [0.0 - 1.0]
        """
        # Initialize novelty tracking
        if not hasattr(self, 'state_visitation'):
            self.state_visitation = defaultdict(int)
            self.state_recency = deque(maxlen=1000)
            self.state_clusters = defaultdict(int)
            self.state_recency_map = {}
        
        # 1. Generate state fingerprint
        state_fingerprint = self._generate_state_fingerprint(state)
        
        # 2. Frequency-based novelty
        visit_count = self.state_visitation[state_fingerprint] + 1
        frequency_novelty = 1.0 / math.sqrt(visit_count)
        
        # 3. Temporal recency (hours since last seen)
        current_time = time.time()
        last_seen = self.state_recency_map.get(state_fingerprint, 0)
        temporal_novelty = 0.0
        if last_seen > 0:
            hours_since_seen = (current_time - last_seen) / 3600
            temporal_novelty = min(1.0, 0.5 / (1 + math.exp(-hours_since_seen/12)))
        
        # 4. Embedding cluster density
        if hasattr(self, 'state_encoder'):
            state_embedding = self.state_encoder(state)
            cluster_id = self._assign_to_cluster(state_embedding)
            cluster_density = self.state_clusters[cluster_id]
            density_novelty = 1.0 - math.tanh(cluster_density / 50)
        else:
            density_novelty = 0.5
        
        # 5. Prediction uncertainty
        if hasattr(self, 'dynamics_model'):
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                pred_variance = self.dynamics_model.forward_variance(state_tensor).item()
                uncertainty_novelty = min(1.0, pred_variance * 5)
        else:
            uncertainty_novelty = 0.3
        
        # Combine factors with adaptive weights
        weights = np.array([0.4, 0.2, 0.2, 0.2])  # Frequency, Temporal, Density, Uncertainty
        novelty_score = np.dot(weights, [frequency_novelty, temporal_novelty, density_novelty, uncertainty_novelty])
        
        # Update state tracking
        self.state_visitation[state_fingerprint] = visit_count
        self.state_recency.append(state_fingerprint)
        self.state_recency_map[state_fingerprint] = current_time
        
        # Dynamic normalization
        max_novelty = max(1.0, np.max(list(self.state_visitation.values())) / 10)
        return np.clip(novelty_score / max_novelty, 0.0, 1.0) * self._curiosity_scaling_factor()
    
    def _calculate_consistency(self, state, action):
        """
        Reward for consistent behavior in similar states:
        1. Action consistency in similar states
        2. Temporal consistency in state transitions
        """
        # 1. Action consistency in similar states
        state_embedding = self._process_state(state)
        similar_states = self.learning_factory.get_similar_states(state_embedding, k=5)
        
        if not similar_states:
            return 0.0
            
        similar_actions = [self.action_memory[s] for s in similar_states]
        action_consistency = 1.0 if action in similar_actions else -0.2
        
        # 2. Temporal consistency
        temporal_consistency = 0.0
        if len(self.state_history) >= 2:
            prev_state = self.state_history[-2]
            prev_action = self.action_history[-2]
            transition_prob = self.transition_model.get_probability(prev_state, prev_action, state)
            temporal_consistency = min(1.0, transition_prob * 2)  # Scale probability to [0, 1]
        
        return 0.6 * action_consistency + 0.4 * temporal_consistency

    def _calculate_energy_cost(self, action):
        """
        Penalty for computationally expensive actions:
        1. Model-based action cost
        2. Memory access cost
        3. Communication cost
        """
        # 1. Model-based action cost
        if action in self.complex_actions:
            cost = 0.2  # Higher cost for model-based actions
        else:
            cost = 0.05  # Lower cost for simple actions
            
        # 2. Memory access cost
        memory_cost = min(0.1, len(self.learning_factory.state_history) / 100000)
        
        # 3. Communication cost (if applicable)
        comm_cost = 0.0
        if action in self.communication_actions:
            comm_cost = 0.15
            
        return cost + memory_cost + comm_cost

    @property
    def complex_actions(self):
        """
        Returns a list of action indices considered computationally or behaviorally complex.
        These typically incur higher energy cost or involve multi-step execution.
        """
        return self._config.get("complex_actions", [0, 2, 4])
    
    @property
    def communication_actions(self):
        """
        Returns a list of action indices dedicated to inter-agent communication.
        These may involve signaling, broadcasting, or transmitting information.
        """
        return self._config.get("communication_actions", [1, 3, 5])

    def _is_unsafe(self, action: int) -> bool:
        """
        Check if an action is unsafe using SafetyGuard.
        
        Args:
            action: Action index to validate
            
        Returns:
            True if action is unsafe, False otherwise
            
        Raises:
            ToxicContentError if action triggers safety violation
        """
        # Convert action to semantic representation for safety validation
        action_description = self.env.action_space.actions[action] if hasattr(self.env.action_space, 'actions') else f"Action_{action}"
        
        try:
            # Initialize SafetyGuard if needed
            if not hasattr(self, 'safety_guard'):
                from src.agents.safety.safety_guard import SafetyGuard
                from src.agents.safety.utils.security_error import ToxicContentError, PrivacyViolationError, PiiLeakageError
                self.safety_guard = SafetyGuard()
            
            # Create safety context
            context = {
                "current_state": self._current_state.tolist() if hasattr(self, '_current_state') else [],
                "action_description": action_description,
                "episode": self._episode_count,
                "step": self._step_count
            }
            
            # Validate action through safety guard
            validation = self.safety_guard.validate_action({
                'state': context,
                'proposed_action': action_description,
                'context': context
            })
            
            return not validation['approved']
        
        except (ToxicContentError, PrivacyViolationError, PiiLeakageError) as e:
            self.logger.error(f"Unsafe action detected: {action} - {str(e)}")
            return True
        except Exception as e:
            self.logger.warning(f"Safety check failed: {str(e)}")
            return False
    
    def _generate_state_fingerprint(self, state):
        """Create unique state identifier for tracking"""
        if isinstance(state, torch.Tensor):
            return hash(state.cpu().numpy().tobytes())
        return hash(str(state))
    
    def _curiosity_scaling_factor(self):
        """Adaptive scaling based on learning stage"""
        progress = self.shared_memory.get('training_progress', 0.0)
        return 0.3 * (1 + np.sin(progress * np.pi / 0.5))

    def _run_rsi_self_improvement(self):
        """Enhanced RSI integration with architectural evolution"""
        # 1. Performance analysis
        performance_report = self.rsi_agent.analyze_performance(
            self.performance_history,
            self.architecture_snapshot
        )
        
        # 2. Architecture optimization
        if performance_report['architecture_update_needed']:
            new_architecture = self.rsi_agent.propose_architecture(
                self.architecture_history,
                performance_report
            )
            self._evolve_network_architecture(new_architecture)
            
        # 3. Hyperparameter optimization
        optimized_params = self.rsi_agent.optimize_hyperparameters(
            self.performance_tracker,
            self.config
        )
        self._update_agent_hyperparameters(optimized_params)
        
        # 4. Knowledge distillation
        self.rsi_agent.distill_knowledge(
            teacher_agents=self.agents,
            student_agent=self.rsi_agent
        )

    def _evolve_network_architecture(self, new_arch):
        """Dynamic network architecture modification"""
        # Capture snapshot BEFORE updating architecture
        architecture_snapshot = {}
        for agent_id in ['dqn', 'maml']:
            agent = self.agents[agent_id]
            if hasattr(agent, 'policy_net'):
                architecture_snapshot[agent_id] = {
                    'policy_net': copy.deepcopy(agent.policy_net),
                    'target_net': copy.deepcopy(agent.target_net) if hasattr(agent, 'target_net') else None
                }
        
        # Save snapshot to history
        self.architecture_history.append(architecture_snapshot)
        
        # Proceed with architecture update
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
            'current_state': self.state_history[-1] if self.state_history else None,
            'last_action': self.action_history[-1] if self.action_history else None,
            'learning_phase': self.learning_phase,
            'active_strategy': self.active_strategy,
            'epsilon': self.agents[self.active_strategy].epsilon if hasattr(self.agents[self.active_strategy], 'epsilon') else None,
            'learning_rate': self.agents[self.active_strategy].learning_rate if hasattr(self.agents[self.active_strategy], 'learning_rate') else None,
            'gradient_norm': self._get_gradient_norm(self.agents[self.active_strategy]),
            'recent_loss': self.performance_metrics['loss'][-10:] if 'loss' in self.performance_metrics else [],
            'recent_rewards': self.reward_history[-10:]
        }

    def _get_gradient_norm(self, agent):
        """Safety check for gradient stability"""
        if hasattr(agent, 'policy_net'):
            params = [p.grad for p in agent.policy_net.parameters() if p.grad is not None]
            return torch.norm(torch.stack([torch.norm(p) for p in params])).item()
        return 0.0

    def _recover_soft_reset(self):
        """Level 1 Recovery: Reset network weights and clear buffers"""
        logger.warning("Performing soft reset")
        # Reset neural network weights
        for agent in self.agents.values():
            if hasattr(agent, 'policy_net'):
                agent.policy_net.reset_parameters()
                
        # Clear experience buffers
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        # Reset exploration parameters
        for agent_id in self.agents:
            if hasattr(self.agents[agent_id], 'epsilon'):
                self.agents[agent_id].epsilon = min(1.0, self.agents[agent_id].epsilon * 1.5)
                
        return {'status': 'recovered', 'strategy': 'soft_reset'}

    def _recover_learning_rate_adjustment(self):
        """Level 2 Recovery: Adaptive learning rate scaling"""
        logger.warning("Adjusting learning rates")
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'learning_rate'):
                agent.learning_rate *= 0.5
                agent.learning_rate = max(agent.learning_rate, 1e-6)
        return {'status': 'recovered', 'strategy': 'lr_adjustment'}

    def _recover_strategy_switch(self):
        """Level 3 Recovery: Fallback to safe strategy"""
        logger.warning("Switching to safe strategy")
        self.active_strategy = 'rl'  # Default to basic RL
        if self.safety_guard:
            return self.safety_guard.execute({'task': 'emergency_override'})
        return {'status': 'recovered', 'strategy': 'strategy_switch'}

    def _full_reset(self):
        """Reinitialize core components while preserving environment"""
        # Preserve essential references
        env = self.env
        config = self.config
        shared_memory = self.shared_memory
        agent_factory = self.agent_factory
        
        # Reinitialize main components
        self.__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            env=env,
            config=config
        )

        # Initialize sub-agents
    def _initialize_agents(self, env, performance_metrics, 
                 mutation_rate=0.1, top_k=2):
        self.env = env
        self.performance = performance_metrics
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
                agent_id=agent_id
            )
        elif agent_id == 'maml':
            return MAMLAgent(
                state_size=state_size,
                action_size=action_size,
                agent_id=agent_id
            )
        elif agent_id == 'rsi':
            return RSIAgent(
                state_size=state_size,
                action_size=action_size,
                agent_id=agent_id
            )
        elif agent_id == 'rl':
            return RLAgent(
                possible_actions=list(range(action_size)),
                state_size=state_size,
                agent_id=agent_id
            )
        raise ValueError(f"Unknown agent type: {agent_id}")

    def _select_action(self, state, agent_type):
        """ Safe action selection """
        action = super()._select_action(state, agent_type)

        # Validate with safety agent
        validation = self.safety_guard.validate_action({
            'state': state,
            'proposed_action': action,
            'agent_type': agent_type
        })

        if not validation['approved']:
            self.logger.warning(f"Unsafe action {action} detected, using corrected action")
            return validation['corrected_action']
            
        return action

    def _process_state(self, raw_state):
        return self.state_processor.process(raw_state)

    def _training_error_handler(func):
        """Decorator for error recovery in training methods"""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (NaNException, GradientExplosionError, InvalidActionError) as e:
                self.logger.error(f"Training error: {str(e)}")
                
                # Use recovery system instead of old error handling
                self.recovery_system.increment_error_count()
                recovery_result = self.recovery_system.execute_recovery()
                
                # Reset error count if recovery was successful
                if recovery_result.get('status') == 'recovered':
                    self.recovery_system.reset_error_count()
                
                return recovery_result
        return wrapper

    def train_step(self, state, action, reward, next_state, task_id):
        """
        Robust training step with dynamic input handling and gradient safety
        """
        try:
            # 1. Convert inputs to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            
            # 2. Generate proper embeddings
            task_embedding = self._generate_task_embedding_(state_tensor)
            
            # 3. Dynamic resizing if needed
            if task_embedding.shape[-1] != self.task_embedding_dim:
                self._resize_embedder(task_embedding.shape[-1])
                task_embedding = self.state_embedder(state_tensor.unsqueeze(0))
            
            # 4. Forward pass with gradient monitoring
            self.optimizer.zero_grad()
            logits = self.policy_net(task_embedding)
            
            # 5. Compute loss with numerical stability checks
            loss = self._compute_task_loss(
                prediction=logits,
                action=action,
                reward=reward_tensor,
                next_state=next_state_tensor,
                task_id=task_id
            )
            
            # 6. Backpropagation with gradient clipping
            if not loss.requires_grad:
                logger.warning("Loss doesn't require gradients, skipping backward")
                return 0.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            return loss.item()
            
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # Handle input dimension mismatch dynamically
                self._adapt_to_input_change(state_tensor)
                return 0.0  # Skip this step after adaptation
            else:
                raise e
        except Exception as e:
            logger.error(f"Train step failed: {e}")
            return 0.0

    def _adapt_to_input_change(self, state_tensor):
        """Dynamically adjust network architecture for changed input dimensions"""
        new_dim = state_tensor.shape[-1]
        logger.warning(f"Input dimension changed to {new_dim}, adapting networks...")
        
        # Resize state embedder
        self.state_dim = new_dim
        self.state_embedder = nn.Sequential(
            nn.Linear(new_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.task_embedding_dim)
        ).to(self.device)
        
        # Reinitialize policy network
        hidden_dim = self.learning_config.get('meta_controller', {}).get('hidden_dim', 128)
        self.policy_net = nn.Sequential(
            nn.Linear(self.task_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_agent_strategies)
        ).to(self.device)
        
        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_config.get('meta_controller', {}).get('learning_rate', 1e-3)
        )

    def _compute_task_loss(self, prediction, action, reward, next_state, task_id):
        """Safe loss calculation with fallbacks"""
        try:
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(0)

            if isinstance(action, int):
                action = torch.tensor([action], dtype=torch.long)

            # Get task metadata
            task_type = self._get_task_type(task_id)
            
            if task_type == "rl":
                # Use agent's Q-network instead of policy_net
                with torch.no_grad():
                    next_q = self.agents['dqn'].policy_net(next_state.unsqueeze(0))
                    target = reward + self.agents['dqn'].gamma * torch.max(next_q)
                current_q = prediction[0, action]
                return torch.nn.functional.mse_loss(current_q, target)
                
            else:
                return torch.nn.functional.cross_entropy(prediction, action)
                
        except Exception as e:
            self.logger.error(f"Loss computation failed: {e}")
            return torch.tensor(0.0, requires_grad=True)

    def _generate_task_embedding_(self, state):
        """Robust embedding generation with shape handling"""
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        return self.state_embedder(state).squeeze(0)

    def _get_task_type(self, task_id):
        """Determine task type from configuration or task ID pattern"""
        # In real implementation, this would come from config
        if "classify" in task_id:
            return "classification"
        elif "regress" in task_id:
            return "regression"
        else:
            return "rl"

    @_training_error_handler
    def train_episode(self, agent_type="dqn"):
        """
        Execute one training episode with selected strategy
        
        Implements Hybrid Reward Architecture from:
        van Seijen et al. (2017). Hybrid Reward Architecture for RL
        """
        try:
            state = self._process_state(self.env.reset())
            action = self._select_action(state, agent_type)
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
                next_state, env_reward, done, _ = self.env.step(action)
                processed_next_state = self._process_state(next_state)

                # Calculate enhanced reward
                enhanced_reward = self._calculate_reward(
                    state, action, processed_next_state, env_reward
                )

                # Store experience with original reward
                episode_data.append((state, action, env_reward, processed_next_state, done))
                self._process_feedback(state, action, enhanced_reward)

                total_reward += enhanced_reward
                state = processed_next_state
                if done: break

            # Train on episode data
            loss = self._train_strategy(episode_data, agent_type)
            self.performance_history.append(total_reward)
            self.performance_metrics['dqn_loss'].append(dqn_loss)
            self.performance_metrics['maml_grad_norm'].append(grad_norm)

            # Update strategy weights
            self._update_strategy_weights(loss)

            # Rebalance after each episode
            self.multi_task_learner.rebalance()

            return total_reward
        finally:
            if not has_error:
                self.recovery_system.reset_error_count()

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
        if gradients is None:
            return "No gradients found"

        gradients = self._scale_by_dendritic_branches(gradients)    # Dendritic compartmentalization
        gradients = self._apply_glial_modulation(gradients)         # Glial-guided learning
        return self._stdp_weight_update(gradients)                  # STDP-based weight update

    def _scale_by_dendritic_branches(self, gradients):
        """Simulate dendritic tree computation through gradient modulation"""
        for param, grad in gradients.items():
            if 'hidden' in param.name:
                gradients[param] *= self.dendritic_scaling * self.glial_support_factor
        return gradients
    
    def _apply_glial_modulation(self, gradients):
        """Glial-guided learning through neuromodulation"""
        # 1. Calculate metabolic support level
        metabolic_support = self._calculate_metabolic_support()
        
        # 2. Apply glial modulation to gradients
        for param, grad in gradients.items():
            if 'excitatory' in param.name:
                gradients[param] *= self.synaptic_strengths['excitatory'] * metabolic_support
            elif 'inhibitory' in param.name:
                gradients[param] *= self.synaptic_strengths['inhibitory'] * metabolic_support
            else:
                gradients[param] *= self.synaptic_strengths['modulatory'] * metabolic_support
                
        return gradients

    def _stdp_weight_update(self, gradients):
        """Spike-Timing Dependent Plasticity weight update"""
        # Only apply STDP to excitatory synapses
        for param, grad in gradients.items():
            if 'excitatory' in param.name:
                # Get pre-post spike timing difference from buffer
                if self.stdp_window:
                    avg_timing_diff = np.mean([d['timing_diff'] for d in self.stdp_window])
                    
                    # Apply STDP rule
                    if avg_timing_diff > 0:  # Pre-before-post (LTP)
                        stdp_factor = self.stdp_parameters['A_plus'] * math.exp(-avg_timing_diff/self.stdp_parameters['tau_plus'])
                    else:  # Post-before-pre (LTD)
                        stdp_factor = -self.stdp_parameters['A_minus'] * math.exp(avg_timing_diff/self.stdp_parameters['tau_minus'])
                        
                    gradients[param] += stdp_factor * param.data
                    
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

            # Evolutionary optimization step
            if ep % 500 == 0:
                strategy_performance = self._evaluate_strategies()
                optimized_agents = self._evolve_strategies(strategy_performance)
                self._update_agent_pool(optimized_agents)
                self.learning_factory.save_checkpoints()

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
            self.learning_factory._replay_historical_data()

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

    def _check_learning_triggers(self):
        """Evaluate activation conditions for learning"""
        return any([
            self.learning_factory._detect_new_data(),
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
        """Use LearningFactory's selection logic as primary strategy"""
        task_metadata = {
            'complexity': self._calc_env_complexity(),
            'performance_history': self.performance_history
        }
        selected = self.learning_factory.select_agent(task_metadata)
        return selected.agent_id.split('_')[0]  # Extract base type

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

    def _requires_meta_update(self):
        """Check if meta-update should be performed"""
        # Check based on performance plateau
        recent_perf = list(self.performance_history)[-10:]
        if len(recent_perf) < 5:
            return False
            
        plateau_threshold = 0.05  # 5% change considered plateau
        max_perf, min_perf = max(recent_perf), min(recent_perf)
        if (max_perf - min_perf) / max_perf < plateau_threshold:
            return True
            
        # Check based on novelty of recent tasks
        if self.concept_drift_detector.detected_drift():
            return True
            
        # Check scheduled update
        if self.meta_update_counter >= self.meta_update_interval:
            return True
            
        return False

    def _evaluate_strategies(self):
        """Assess all learning strategies using validation tasks"""
        return {
            'dqn': self.agents['dqn'].evaluate(self.env),
            'maml': self.agents['maml'].evaluate(env=self.env),
            'rsi': self.agents['rsi'].evaluate(self.env),
            'rl': self.agents['rl'].evaluate(self.env)
        }

    def _apply_ewc_regularization(self):
        """Implement Elastic Weight Consolidation"""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'consolidate_weights'):
                agent.consolidate_weights(self.ewc_lambda)

    def _evolve_strategies(self, performance_data):
        """Use LearningFactory to generate optimized agents"""
        self.learning_factory.performance = performance_data
        return self.learning_factory.generate_new_strategies()

    def _update_agent_pool(self, optimized_agents):
        """Replace underperforming agents with evolved versions"""
        for agent in optimized_agents:
            agent_type = self._detect_agent_type(agent)
            if agent_type in self.agents:
                current_score = np.mean(self.performance_metrics[agent_type][-50:] or 0)
                new_score = agent.evaluate(env)
                if new_score > current_score:
                    self.agents[agent_type] = agent
                    self.logger.info(f"Upgraded {agent_type} with evolved version")

    def _detect_agent_type(self, agent):
        """Determine agent type from instance"""
        if isinstance(agent, DQNAgent):
            return 'dqn'
        elif isinstance(agent, MAMLAgent):
            return 'maml'
        elif isinstance(agent, RSIAgent):
            return 'rsi'
        elif isinstance(agent, RLAgent):
            return 'rl'
        else:
            # Check for hybrid agents
            if hasattr(agent, 'agent_type'):
                return agent.agent_type
            return 'unknown'

    def _run_rsi_self_improvement(self):
        """RSI optimization cycle"""
        analysis = self.agents["rsi"].execute({
            "performance_history": self.performance_history,
            "strategy_weights": self.strategy_weights
        })
        logger.info(f"RSI optimization result: {analysis}")

    def _adapt_exploration_strategy(self):
        """
        Dynamically adjust exploration parameters based on learning progress.
        
        Implements:
        - Epsilon decay based on performance
        - Curiosity-driven exploration boosting
        - Uncertainty-guided exploration
        """
        # Performance-based exploration adjustment
        if len(self.performance_history) > 20:
            recent_perf = np.mean(self.performance_history[-10:])
            baseline_perf = np.mean(self.performance_history[:10])
            
            # Calculate performance improvement ratio
            improvement_ratio = (recent_perf - baseline_perf) / (abs(baseline_perf) + 1e-6)
            
            # Adjust exploration based on improvement
            if improvement_ratio > 0.1:  # Good improvement
                decay_factor = 0.95  # Reduce exploration
            elif improvement_ratio < -0.1:  # Performance degradation
                decay_factor = 1.05  # Increase exploration
            else:  # Stable performance
                decay_factor = 0.99  # Slow decay
            
            # Apply to all agents with epsilon
            for agent in self.agents.values():
                if hasattr(agent, 'epsilon'):
                    new_epsilon = agent.epsilon * decay_factor
                    agent.epsilon = max(0.01, min(1.0, new_epsilon))
        
        # Curiosity-driven exploration boost
        if hasattr(self, 'novelty_threshold'):
            if len(self.embedding_buffer) > 50:
                embeddings = [e[0] for e in self.embedding_buffer]
                avg_novelty = np.mean([self._calculate_novelty_bonus(e) for e in embeddings])
                
                if avg_novelty < self.novelty_threshold:
                    for agent in self.agents.values():
                        if hasattr(agent, 'epsilon'):
                            agent.epsilon = min(1.0, agent.epsilon * 1.2)
        
        # Uncertainty-guided exploration
        if hasattr(self, 'uncertainty_threshold') and hasattr(self, 'dynamics_model'):
            state = self._current_state if hasattr(self, '_current_state') else None
            if state is not None:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    uncertainty = self.dynamics_model.forward_variance(state_tensor).item()
                    
                if uncertainty > self.uncertainty_threshold:
                    for agent in self.agents.values():
                        if hasattr(agent, 'epsilon'):
                            agent.epsilon = min(1.0, agent.epsilon * 1.3)
        
        # Log exploration strategy update
        epsilons = [agent.epsilon for agent in self.agents.values() if hasattr(agent, 'epsilon')]
        if epsilons:
            avg_epsilon = np.mean(epsilons)
            self.logger.info(f"Exploration strategy updated - Avg epsilon: {avg_epsilon:.3f}")

    def _update_strategy_weights(self, rewards):
        strategy_performance = {
            agent_id: np.mean(rewards[agent_id][-10:]) 
            for agent_id in self.agents
        }
        total = sum(strategy_performance.values())
        self.strategy_weights = np.array([strategy_performance[id]/total for id in self.agents.keys()])

    def _check_performance_drop(self):
        """Calculate relative performance degradation"""
        reward_data = list(self.performance_metrics['reward'])
        if not reward_data:
            return False
            
        recent_perf = np.mean(reward_data[-100:])
        max_perf = np.max(reward_data)
        return recent_perf < (self.performance_threshold * max_perf)

    def _detect_concept_drift(self):
        """Statistical test for distribution shifts using KL-divergence"""
        # Get recent state distribution
        recent_states = self.state_history[-100:]
        if len(recent_states) < 20:
            return False
            
        # Compare with historical distribution
        historical_states = self.state_history[-1000:-100]
        if len(historical_states) < 50:
            return False
            
        # Calculate KL-divergence
        kl_div = self.learning_calculations._calculate_kl_divergence(historical_states, recent_states)
        return kl_div > self.data_change_threshold

    def _check_scheduled_retraining(self):
        """Time-based retraining trigger using TaskScheduler
        
        Implements intelligent scheduling considering:
        - Time since last retraining
        - Current system load
        - Performance trends
        - Resource availability
        """
        # Initialize scheduler if not already done
        if not hasattr(self, '_task_scheduler'):
            from src.agents.planning.task_scheduler import DeadlineAwareScheduler
            self._task_scheduler = DeadlineAwareScheduler()
        
        # Create retraining task
        retraining_task = {
            'id': 'periodic_retraining',
            'requirements': ['retraining', 'model_update'],
            'deadline': time.time() + self.retraining_interval.total_seconds(),
            'priority': 2,  # Medium priority
            'metadata': {
                'last_retraining': self.last_retraining.timestamp(),
                'performance_trend': self._calculate_performance_trend(),
                'resource_usage': self._get_current_resource_usage()
            }
        }
        
        # Create virtual agent representing our retraining capability
        retraining_agent = {
            'self_retraining': {
                'capabilities': ['retraining', 'model_update'],
                'current_load': self._calculate_learning_load(),
                'efficiency': self._get_retraining_efficiency(),
                'successes': len([x for x in self.performance_history if x > 0]),
                'failures': len([x for x in self.performance_history if x <= 0])
            }
        }
        
        # Schedule the retraining task
        schedule = self._task_scheduler.schedule(
            tasks=[retraining_task],
            agents=retraining_agent,
            risk_assessor=self._assess_retraining_risk,
            state={
                'tasks': [retraining_task],
                'dependency_graph': self._get_retraining_dependencies()
            }
        )
        
        # Check if scheduling approved the retraining now
        if schedule.get('periodic_retraining', {}).get('start_time', float('inf')) <= time.time():
            self.last_retraining = datetime.now()
            return True
        
        return False
    
    def _calculate_performance_trend(self):
        """Calculate performance trend over last 100 episodes"""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = np.mean(self.performance_history[-10:])
        baseline = np.mean(self.performance_history[:10])
        return (recent - baseline) / (abs(baseline) + 1e-6)
    
    def _get_current_resource_usage(self):
        """Get current system resource usage"""
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'gpu': self._get_gpu_usage() if torch.cuda.is_available() else 0.0
        }
    
    def _calculate_learning_load(self):
        """Calculate current learning system load (0-1 scale)"""
        active_processes = sum(1 for _ in self.agents.values() if hasattr(_, 'is_training') and _.is_training)
        max_concurrent = self.learning_config.get('max_concurrent_training', 3)
        return min(1.0, active_processes / max_concurrent)
    
    def _get_retraining_efficiency(self):
        """Calculate retraining efficiency based on historical performance"""
        if not hasattr(self, '_retraining_history'):
            return 1.0
        
        improvements = []
        for i in range(1, len(self._retraining_history)):
            prev = self._retraining_history[i-1]['performance']
            curr = self._retraining_history[i]['performance']
            improvements.append((curr - prev) / (abs(prev) + 1e-6))
        
        return np.mean(improvements) if improvements else 1.0
    
    def _assess_retraining_risk(self, task):
        """Risk assessment for retraining task"""
        volatility = np.std(list(self.performance_history)[-50:] or [0])
        stability_score = 1.0 / (volatility + 1e-6)
        
        return {
            'risk_score': min(1.0, 0.7 - 0.5 * stability_score),
            'recommendations': [
                'reduce_batch_size' if volatility > 0.5 else 'proceed',
                'warm_start' if len(self.performance_history) > 100 else 'cold_start'
            ]
        }
    
    def _get_retraining_dependencies(self):
        """Get dependencies for retraining tasks"""
        deps = defaultdict(list)
        
        # Add dependencies based on current learning state
        if hasattr(self, 'active_strategy'):
            deps['strategy_selection'] = ['periodic_retraining']
        
        if len(self.performance_history) > 50:
            deps['performance_analysis'] = ['periodic_retraining']
        
        return deps

    # Core Learning Components
    def _run_meta_learning_phase(self):
        """Execute MAML-based meta-learning"""
        task_pool = self._generate_meta_tasks()
        meta_loss = self.agents['maml'].meta_update(task_pool)
        self.performance_metrics['meta_loss'].append(meta_loss)

    def _generate_meta_tasks(self):
        """Create task variations for meta-learning"""
        tasks = []
        
        # 1. Physics-based variations
        for _ in range(2):
            env_variant = self.env._create_physics_variation()
            tasks.append((env_variant, {'variation_type': 'physics'}))
            
        # 2. Reward-shaping variations
        for _ in range(2):
            env_variant = self._create_reward_variation()
            tasks.append((env_variant, {'variation_type': 'reward'}))
            
        # 3. Observation variations
        tasks.append((self._add_observation_noise(), {'variation_type': 'sensory'}))
        
        # 4. Adversarial perturbations
        tasks.append((self._add_adversarial_perturbation(), {'variation_type': 'adversarial'}))
        
        return tasks

    def _create_reward_variation(self):
        """Create reward-shaping variation"""
        env_variant = copy.deepcopy(self.env)
        if hasattr(env_variant.unwrapped, 'reward_weights'):
            original_weights = env_variant.unwrapped.reward_weights
            new_weights = {k: v * np.random.uniform(0.5, 1.5) for k, v in original_weights.items()}
            env_variant.unwrapped.reward_weights = new_weights
        return env_variant

    def _add_adversarial_perturbation(self):
        """Add adversarial perturbations to observations"""
        class AdversarialWrapper(gym.Wrapper):
            def __init__(self, env, perturbation_strength=0.1):
                super().__init__(env)
                self.perturbation_strength = perturbation_strength
                
            def step(self, action):
                obs, reward, done, truncated, info = self.env.step(action)
                # Add perturbation in the direction that minimizes Q-value
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    obs_tensor.requires_grad = True
                    q_values = self.agent.policy_net(obs_tensor.unsqueeze(0))
                    target_q = torch.min(q_values)
                    target_q.backward()
                    perturbation = self.perturbation_strength * torch.sign(obs_tensor.grad)
                return (obs + perturbation.numpy(), reward, done, truncated, info)
                
        return AdversarialWrapper(self.env)

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
            logger.info(f"Created task variant with params: {params}")
            
            return env_variant
            
        except Exception as e:
            logger.warning(f"Task variation failed: {str(e)}")
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

    def extract_new_rules(self):
        return []

    # For a financial advisor
    def get_learning_status(self):
        """Return current learning status"""
        return {
            "training_mode": self.training_mode,
            "performance_metrics": self.performance_metrics,
            "performance_metrics": self.metric_store.get_metrics_summary() 
            if isinstance(self.strategy_weights, np.ndarray) 
            else self.strategy_weights
        }
    
    def predict(self, state: Any = None) -> Dict[str, Any]:
        """
        Predicts an action using the meta-controller to select the best strategy,
        then delegates to the selected agent's prediction mechanism.
        
        Returns structured prediction output containing:
            - selected_strategy: The chosen learning strategy
            - action: The predicted action
            - confidence: Confidence score of the prediction
            - strategy_output: Raw output from the selected strategy agent
        """
        # Handle missing state
        if state is None:
            if hasattr(self, '_current_state'):
                state = self._current_state
            else:
                logger.warning("No state provided for prediction")
                return {
                    "selected_strategy": "unknown",
                    "action": 0,
                    "confidence": 0.0,
                    "strategy_output": None
                }
        
        try:
            # Generate state embedding
            state_embedding = self._generate_task_embedding_(state)
            
            # Select best strategy using meta-controller
            strategy = self.select_agent_strategy(state_embedding)
            agent = self.agents.get(strategy)
            
            if not agent:
                raise ValueError(f"Strategy '{strategy}' not found in agent registry")
            
            # Get prediction from selected agent
            if hasattr(agent, 'predict'):
                result = agent.predict(state)
            elif hasattr(agent, 'get_action'):
                result = agent.get_action(state, explore=False)
            else:
                result = agent.act(state) if hasattr(agent, 'act') else 0
            
            return {
                "selected_strategy": strategy,
                "action": result if isinstance(result, int) else result.get('action', 0),
                "confidence": 1.0,  # Placeholder for actual confidence
                "strategy_output": result
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "selected_strategy": "error",
                "action": 0,
                "confidence": 0.0,
                "strategy_output": None
            }

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
        self.learning_calculations = LearningCalculations()
        kl_div = self.learning_calculations._calculate_kl_divergence(historical_data, current_batch)
        self.data_window.extend(current_batch)
        
        return kl_div > self.threshold

__all__ = ['LearningAgent', 'SLAIEnv']

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Learning Agent ===\n")
    shared_memory = {}
    agent_factory = lambda: None
    env = SLAIEnv(state_dim=4, action_dim=2, max_steps=500)

    agent = LearningAgent(shared_memory, agent_factory, env=env, config=None)

    print(f"\n{agent}\n")

    print("\n* * * * * Phase 2 * * * * *\n")
    embedding_with_label=None
    best_agent_strategy_name=[]
    observe = agent.observe(embedding_with_label, best_agent_strategy_name)
    print(f"\n{observe}")

    print("\n* * * * * Phase 3 * * * * *\n")
    state, _ = env.reset()
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    task_id = 'default_task'
    
    # Pass real state instead of dummy embedding
    train = agent.train_step(state, action, reward, next_state, task_id)
    
    print(f"\nTraining step result: {train}")

    print("\n* * * * * Phase 4 * * * * *\n")
    done = terminated or truncated
    calculate = agent._calculate_reward(state, action, next_state, reward)
    print(agent._check_scheduled_retraining())
    print(f"Calculated reward: \n{calculate}")

    print("\n* * * * * Phase 5 * * * * *\n")

    print(agent._execute_continual_learning())
    print("\n=== Successfully Ran Learning Agent ===\n")
