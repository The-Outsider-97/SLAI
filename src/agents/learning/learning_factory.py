
import os
import yaml
import torch
import numpy as np

from collections import defaultdict, deque
from typing import Counter

from src.agents.learning.dqn import DQNAgent
from src.agents.learning.maml_rl import MAMLAgent
from src.agents.learning.rsi import RSIAgent
from src.agents.learning.rl_agent import RLAgent
from src.agents.learning.learning_memory import LearningMemory
from logs.logger import get_logger

logger = get_logger("Leaning Factory")

CONFIG_PATH = "src/agents/learning/configs/learning_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class LearningFactory:
    """Evolutionary strategy optimization factory with parameter mutation"""
    
    def __init__(self, env, performance_metrics=None, config=None):
        if env is None or not hasattr(env, 'observation_space') or not hasattr(env, 'action_space'):
            raise ValueError("LearningFactory requires valid environment with observation_space and action_space")
        self.env = env
        self.performance = performance_metrics or {}
        base_config = load_config() if config is None else config
        self.config = base_config.get('evolutionary', {})
        
        # Load parameters from config
        self.mutation_rate = self.config.get('mutation_rate')
        self.top_k = self.config.get('top_k')

        self.learning_memory = LearningMemory(config=base_config.get('learning_memory', {}))
        self.model_id = "Learning_Factory"
        self.memory = deque(maxlen=10000)

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

        self._init_component(config)

        self.temporary_agents = {}  # Format: {agent_hash: {'agent': obj, 'use_count': int}}
        self.permanent_agents = ['dqn', 'maml', 'rsi', 'rl']
        self.agent_pool = {name: getattr(self, name) for name in self.permanent_agents}
        self.task_registry = defaultdict(int)  # Track task type frequencies

        # Additional monitoring setup
        self.selection_history = deque(maxlen=500)
        self.architecture_snapshot = {
            'hidden_layers': defaultdict(int),
            'activation_functions': Counter()
        }

        logger.info("Learning Factory has successfully initialized")

    def _init_component(self, config):
        # Get state/action dimensions from environment
        if not hasattr(self.env, 'observation_space') or not hasattr(self.env, 'action_space'):
            raise RuntimeError("Environment missing required attributes for agent initialization")
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        state_size = state_dim
        action_size = action_dim
    
        # Load configurations
        base_config = get_merged_config(config)
        self.base_config = base_config

        # Initialize sub-agents with proper parameters
        self.dqn = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_id="dqn_agent"
        )
    
        self.maml = MAMLAgent(
            state_size=state_size,
            action_size=action_size,
            agent_id="maml_agent",
            config=base_config.get('maml', {})
        )
    
        self.rsi = RSIAgent(
            state_size=state_size,
            action_size=action_size,
            agent_id="rsi_agent",
            config=base_config.get('rsi', {})
        )
    
        self.rl = RLAgent(
            possible_actions=list(range(action_dim)),
            state_size=state_size,
            agent_id="rl_agent",
            config=base_config.get('rl', {})
        )

        # Load checkpoints if available
        checkpoint_dir = base_config.get('learning_memory', {}).get('checkpoint_dir')
        if checkpoint_dir:
            for agent_name in ['dqn', 'maml', 'rsi', 'rl']:
                checkpoint_path = os.path.join(checkpoint_dir, f"{agent_name}_checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    getattr(self, agent_name).load_checkpoint(checkpoint_path)

        # Initialize monitoring system
        self.performance_tracker = {
            'dqn': deque(maxlen=100),
            'maml': deque(maxlen=100),
            'rsi': deque(maxlen=100),
            'rl': deque(maxlen=100)
        }

        logger.info("Sub-agents initialized with state_dim:%s, action_dim:%s", state_dim, action_dim)

    def select_agent(self, task_metadata):
        """
        Enhanced agent selection with checkpoint-aware logic
        ===================================================
        Task novelty or variability: Choose MAMLAgent.

        Environment complexity and size:
            - Small → RLAgent
            - Large → DQNAgent

        Performance stagnation: Choose RSIAgent for self-improvement.

        Training budget and compute limits:
            - Low budget → RLAgent or RSIAgent
            - High budget → DQNAgent or MAMLAgent
        """
        # Get latest performance metrics
        recent_performance = {
            agent: np.mean(perf[-10:]) if perf else 0
            for agent, perf in self.performance_tracker.items()
        }
        
        # Checkpoint-based prioritization
        checkpoint_scores = {
            'dqn': self._get_checkpoint_quality('dqn'),
            'maml': self._get_checkpoint_quality('maml'),
            'rsi': self._get_checkpoint_quality('rsi'),
            'rl': self._get_checkpoint_quality('rl')
        }
        
        # Combine metrics
        combined_scores = {
            agent: 0.7*recent_performance.get(agent, 0) + 0.3*checkpoint_scores.get(agent, 0)
            for agent in ['dqn', 'maml', 'rsi', 'rl']
        }

        selected = max(combined_scores, key=combined_scores.get)
        self.selection_history.append(selected)

        # Check if any existing agent can handle the task
        if max(combined_scores.values()) < self.config.get('creation_threshold', 0.4):
            task_type = self._classify_task(task_metadata)
            return self._create_agent(task_type)

        return getattr(self, selected)

    def _get_checkpoint_quality(self, agent_name):
        """Evaluate checkpoint reliability"""
        checkpoint_path = os.path.join(
            self.config['learning_memory']['checkpoint_dir'],
            f"{agent_name}_checkpoint.pt"
        )
        if not os.path.exists(checkpoint_path):
            return 0
            
        try:
            checkpoint = torch.load(checkpoint_path)
            return min(1.0, checkpoint.get('validation_accuracy', 0))
        except:
            return 0

    def monitor_architecture(self):
        """Track agent architecture details for NeuralNetwork-based agents"""
        for agent_name in ['dqn', 'maml', 'rsi']:
            agent = getattr(self, agent_name)
            if hasattr(agent, 'policy_net'):
                # Assume policy_net is a NeuralNetwork instance
                net = agent.policy_net
                # Track hidden layers and their activation functions
                for i in range(net.num_layers - 1):  # Skip output layer
                    layer_type = f"Linear({net.layer_dims[i]}→{net.layer_dims[i+1]})"
                    self.architecture_snapshot['hidden_layers'][layer_type] += 1
                    if i < len(net.hidden_activations):
                        activation_name = net.hidden_activations[i].__class__.__name__
                        self.architecture_snapshot['activation_functions'][activation_name] += 1
                # Track output activation
                output_activation_name = net.output_activation.__class__.__name__
                self.architecture_snapshot['activation_functions'][output_activation_name] += 1

    def save_checkpoints(self):
        """Periodic checkpointing of all agents"""
        checkpoint_dir = self.config['learning_memory']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for agent_name in ['dqn', 'maml', 'rsi', 'rl']:
            agent = getattr(self, agent_name)
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(
                    os.path.join(checkpoint_dir, f"{agent_name}_checkpoint.pt")
                )

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

    def _create_agent(self, task_signature):
        # Check for existing temporary agent
        for agent_hash, data in self.temporary_agents.items():
            if agent_hash.startswith(str(task_signature)):
                data['use_count'] += 1
                if data['use_count'] > self.config.get('promotion_threshold', 3):
                    self._promote_agent(agent_hash, data['agent'])
                return data['agent']
    
        # Create new agent configuration using evolutionary strategy
        agent_type = self._determine_agent_architecture(task_signature)
        new_agent = self._evolve_new_agent(agent_type, task_signature)
        
        # Store temporary agent
        agent_hash = f"{task_signature}_{hash(new_agent)}"
        self.temporary_agents[agent_hash] = {
            'agent': new_agent,
            'use_count': 1
        }
        
        return new_agent
    
    def _determine_agent_architecture(self, task_signature):
        """Determine agent type based on task signature or default to DQN hybrid."""
        if task_signature is None:
            return 'dqn_hybrid'  # Default to DQN hybrid
        # Add logic to map task_signature to agent_type here
        # Example: return 'maml_hybrid' for meta-learning tasks
        return 'dqn_hybrid'  # Default fallback

    def _evolve_new_agent(self, agent_type, task_signature):
        # Hybrid configuration using best-performing agents
        base_config = self._get_base_config(agent_type)
        evolved_params = {
            'hidden_size': np.random.choice([128, 256, 512]),
            'learning_rate': 10**np.random.uniform(-4, -2),
            'gamma': np.clip(np.random.normal(0.95, 0.03), 0.9, 0.999)
        }
        
        # Create agent based on type
        if agent_type == 'dqn_hybrid':
            return DQNAgent(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.n,
                config={'dqn': evolved_params},
                agent_id=f"temp_dqn_{task_signature}"
            )
        elif agent_type == 'maml_hybrid':
            return MAMLAgent(
                state_size=self.env.observation_space.shape[0],
                action_size=self.env.action_space.n,
                config={'maml': evolved_params},
                agent_id=f"temp_maml_{task_signature}"
            )

    def _get_base_config(self, agent_type):
        """Retrieve the base configuration for the given agent type."""
        # Extract base agent name (e.g., 'dqn' from 'dqn_hybrid')
        base_agent = agent_type.split('_')[0]
        return self.base_config.get(base_agent, {})    

    def _promote_agent(self, agent_hash, agent):
        # Generate unique name based on task signature
        signature = agent_hash.split('_')[0]
        agent_name = f"perm_agent_{signature}"
        
        # Add to permanent agents
        setattr(self, agent_name, agent)
        self.permanent_agents.append(agent_name)
        self.agent_pool[agent_name] = agent
        
        # Update factory components
        self.performance_tracker[agent_name] = deque(maxlen=100)
        del self.temporary_agents[agent_hash]
        
        # Save to learning memory
        self.learning_memory.set(
            key=f"promoted_agents/{agent_name}",
            value=agent.get_config()
        )

    def _crossover(self, agent_id1, agent_id2):
        """Combine parameters from two different agent types"""
        # Validate agent IDs
        if agent_id1 not in self.param_bounds or agent_id2 not in self.param_bounds:
            logger.warning("Invalid agent IDs for crossover. Returning empty configuration.")
            return {}

        common_params = set(self.param_bounds[agent_id1]) & set(self.param_bounds[agent_id2])
        hybrid_params = {}
        for param in common_params:
            if np.random.rand() > 0.5:
                hybrid_params[param] = self.param_bounds[agent_id1][param][1]
            else:
                hybrid_params[param] = self.param_bounds[agent_id2][param][1]
        return hybrid_params

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Learning Factory ===\n")
    import gym

    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
            self.action_space = gym.spaces.Discrete(2)

    config = load_config()
    performance_metrics = {}
    env = MockEnv()

    factory = LearningFactory(env, performance_metrics)

    print(f"\n{factory}")

    print("\n* * * * * Phase 2 * * * * *\n")
    task_signature=None
    train = factory._create_agent(task_signature)
    print(f"\n{train}")

    print("\n* * * * * Phase 3 * * * * *\n")
    agent_id1='dqn'
    agent_id2='maml'
    cross = factory._crossover(agent_id1, agent_id2)
    print(f"\n{cross}")

    print("\n* * * * * Phase 4 * * * * *\n")
    monitor = factory.monitor_architecture()
    print(f"\n{monitor}")

    print("\n=== Successfully Ran Learning Factory ===\n")
