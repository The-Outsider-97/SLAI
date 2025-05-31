
import os
import yaml
import torch
import numpy as np

from collections import defaultdict, deque
from typing import Counter

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.dqn import DQNAgent
from src.agents.learning.maml_rl import MAMLAgent
from src.agents.learning.rsi import RSIAgent
from src.agents.learning.rl_agent import RLAgent
from src.agents.learning.learning_memory import LearningMemory
from logs.logger import get_logger

logger = get_logger("Leaning Factory")

class LearningFactory:
    """Evolutionary strategy optimization factory with parameter mutation"""
    
    def __init__(self, env, performance_metrics=None):
        if env is None or not hasattr(env, 'observation_space') or not hasattr(env, 'action_space'):
            raise ValueError("LearningFactory requires valid environment with observation_space and action_space")
        self.env = env
        self.performance = performance_metrics or {}
        self.config = load_global_config()
        self.factory_config = get_config_section('evolutionary')
        
        # Load parameters from config
        self.mutation_rate = self.factory_config.get('mutation_rate')
        self.top_k = self.factory_config.get('top_k')

        self.learning_memory = LearningMemory()
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

        self._init_component()

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

    def _init_component(self):
        if not hasattr(self.env, 'observation_space') or not hasattr(self.env, 'action_space'):
            raise RuntimeError("Environment missing required attributes for agent initialization")
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        state_size = state_dim
        action_size = action_dim
    
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
        )
    
        self.rsi = RSIAgent(
            state_size=state_size,
            action_size=action_size,
            agent_id="rsi_agent",
        )
    
        self.rl = RLAgent(
            possible_actions=list(range(action_dim)),
            state_size=state_size,
            agent_id="rl_agent",
        )

        checkpoint_dir = self.config.get('learning_memory', {}).get('checkpoint_dir')
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

    def _classify_task(self, task_metadata):
        return task_metadata

    def monitor_architecture(self):
        """Track agent architecture details for NeuralNetwork-based agents"""
        for agent_name in ['dqn', 'maml', 'rsi', 'rl']:
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
        """Apply q-Gaussian mutation with self-adaptation to parameters within defined bounds."""
        params = {}
        for param, (min_val, max_val) in self.param_bounds[agent_id].items():
            # Get current value (midpoint if no agent-specific value exists)
            base_val = (min_val + max_val) / 2
            
            # Self-adapting q-value (initialize at Gaussian)
            q = self.agent_pool[agent_id].get('q_value', 1.0)
            
            # Generate q-Gaussian noise using Box-Muller method
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)
            
            if q == 1:  # Standard Gaussian
                noise = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            else:
                # Generalized Box-Muller for q-Gaussian
                beta = 1 / (3 - q) if q < 3 else 0.5  # Scale parameter
                noise = np.sqrt(-beta * np.log(1 - u1)) * np.cos(2 * np.pi * u2)
            
            # Apply mutation with scaled step size
            param_range = max_val - min_val
            step_size = self.mutation_rate * param_range
            mutated = base_val + step_size * noise
            
            # Adaptive q-value update (Eq 2 from Tinos & Yang)
            tau_alpha = 1 / np.sqrt(len(self.param_bounds[agent_id]))
            q = q * np.exp(tau_alpha * np.random.randn())
            q = np.clip(q, 0.9, 2.5)  # Constrain q-value
            
            # Update parameters
            params[param] = np.clip(mutated, min_val, max_val)
            params['q_value'] = q  # Store for next mutation
            
            # Handle integer parameters
            if param in ['hidden_size', 'batch_size', 'memory_size', 'adaptation_steps']:
                params[param] = int(params[param])
                
        return params

    def _determine_agent_architecture(self, task_signature):
        """Determine agent type based on task signature characteristics"""
        if task_signature is None:
            return 'dqn_hybrid'  # Default
        
        # Extract task characteristics (example metrics - would come from task_signature)
        novelty = task_signature.get('novelty', 0.5)        # 0-1 scale (1 = completely new)
        complexity = task_signature.get('complexity', 0.5)   # 0-1 scale (1 = highly complex)
        volatility = task_signature.get('volatility', 0.5)   # 0-1 scale (1 = highly dynamic)
        compute_budget = task_signature.get('compute_budget', 0.5)  # 0-1 scale (1 = high compute)
        
        # Agent selection matrix
        if novelty > 0.8 and compute_budget > 0.7:    # High novelty + high compute = MAML hybrid (meta-learning)
            return 'maml_hybrid'
        
        elif volatility > 0.7 and complexity > 0.6:    # High volatility + complexity = RSI hybrid (self-improving)
            return 'rsi_hybrid'
        
        elif complexity < 0.4 and compute_budget < 0.6:    # Low complexity + low compute = RL hybrid (efficient)
            return 'rl_hybrid'
        
        elif complexity > 0.7 and compute_budget > 0.6:    # High complexity + high compute = DQN hybrid (deep learning)
            return 'dqn_hybrid'
        
        elif novelty > 0.6:    # Novel tasks benefit from MAML's adaptation capabilities
            return 'maml_hybrid'
        
        elif volatility > 0.6:    # Volatile environments need RSI's self-optimization
            return 'rsi_hybrid'
        
        # Hybrid combinations for specialized cases
        if complexity > 0.7 and volatility > 0.6:    # Complex volatile environments = DQN+RSI hybrid
            return 'dqn_rsi_hybrid'
        
        elif novelty > 0.7 and volatility > 0.6:    # Novel volatile environments = MAML+RSI hybrid
            return 'maml_rsi_hybrid'
        
        elif complexity < 0.3 and novelty > 0.6:    # Simple but novel tasks = RL+MAML hybrid
            return 'rl_maml_hybrid'
        
        # Default to DQN hybrid
        return 'dqn_hybrid'

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
    
    def _evolve_new_agent(self, agent_type, task_signature):
        state_dim = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
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
                state_dim=state_dim,
                action_dim=action_size,
                agent_id=f"temp_dqn_{task_signature}"
            )
        
        elif agent_type == 'maml_hybrid':
            return MAMLAgent(
                state_size=state_dim,
                action_size=action_size,
                agent_id=f"temp_maml_{task_signature}"
            )
        
        elif agent_type == 'rsi_hybrid':
            return RSIAgent(
                state_size=state_dim,
                action_size=action_size,
                agent_id=f"temp_rsi_{task_signature}"
            )
        
        elif agent_type == 'rl_hybrid':
            return RLAgent(
                possible_actions=list(range(action_size)),
                state_size=state_dim,
                agent_id=f"temp_rl_{task_signature}"
            )
        
        # Hybrid combinations
        elif agent_type == 'dqn_rsi_hybrid':
            # Would be a custom combination of DQN and RSI
            # Placeholder - actual implementation would combine components
            return DQNAgent(
                state_dim=state_dim,
                action_dim=action_size,
                agent_id=f"temp_dqn_rsi_{task_signature}"
            )
        
        elif agent_type == 'maml_rsi_hybrid':
            # Placeholder for MAML+RSI combination
            return MAMLAgent(
                state_size=state_dim,
                action_size=action_size,
                agent_id=f"temp_maml_rsi_{task_signature}"
            )
        
        elif agent_type == 'rl_maml_hybrid':
            # Placeholder for RL+MAML combination
            return RLAgent(
                possible_actions=list(range(action_size)),
                state_size=state_dim,
                agent_id=f"temp_rl_maml_{task_signature}"
            )
        
        # Fallback to DQN hybrid
        return DQNAgent(
            state_dim=state_dim,
            action_dim=action_size,
            agent_id=f"temp_dqn_{task_signature}"
        )

    def _get_base_config(self, agent_type):
        """Retrieve the base configuration for the given agent type."""
        # Extract base agent name (e.g., 'dqn' from 'dqn_hybrid')
        base_agent = agent_type.split('_')[0]
        return self.factory_config.get(base_agent, {})    

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
            self.width = 84  # Add needed properties for visualization
            self.height = 84
    
        # Add reset method
        def reset(self):
            return self.observation_space.sample(), {}  # Return state and empty info dict
    
        # Add step method with proper return signature
        def step(self, action):
            return (
                self.observation_space.sample(),  # next_state
                np.random.rand(),                 # reward
                np.random.choice([True, False]),  # terminated
                False,                            # truncated
                {}                                # info
            )
        
        # Add render method for visualization
        def render(self):
            # Create dummy image (gray with random noise)
            img = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            return img

    config = load_global_config()
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

    print("\n* * * * * Phase 5: Hybrid Agent Test * * * * *\n")
    # Create a complex task signature that requires a hybrid agent
    task_signature = {
        'novelty': 0.85,    # Highly novel task
        'complexity': 0.75,  # Complex environment
        'volatility': 0.8,   # Highly dynamic
        'compute_budget': 0.9 # Sufficient compute resources
    }

    # Create a hybrid agent for this task
    hybrid_agent = factory._create_agent(task_signature)
    print(f"Created hybrid agent of type: {type(hybrid_agent).__name__}")

    # Test the hybrid agent in the environment
    state = env.reset()
    print(f"\nInitial state: {state}")

    # Get an action from the hybrid agent
    try:
        # Try different agent interfaces
        if hasattr(hybrid_agent, 'select_action'):
            action = hybrid_agent.select_action(state)
        elif hasattr(hybrid_agent, 'get_action'):
            action, _, _ = hybrid_agent.get_action(state)
        elif hasattr(hybrid_agent, 'act'):
            action = hybrid_agent.act(state)
        elif hasattr(hybrid_agent, 'choose_action'):
            action = hybrid_agent.choose_action(state)
        else:
            action = hybrid_agent.get_action(state)[0]  # Default to MAML interface
        
        print(f"Hybrid agent selected action: {action}")
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")
        
    except Exception as e:
        print(f"Error testing hybrid agent: {str(e)}")

    print("\n=== Hybrid Agent Test Completed ===")

    print("\n=== Successfully Ran Learning Factory ===\n")
