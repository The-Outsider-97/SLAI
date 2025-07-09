"""
Deep Q-Network (DQN) Agent with Evolutionary Hyperparameter Optimization

Key Academic References:
1. DQN & Experience Replay: 
   Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
2. Target Networks: 
   Mnih et al. (2015) [Same as above].
3. Evolutionary Strategies: 
   Salimans et al. (2017). Evolution Strategies as a Scalable Alternative to RL. arXiv.

Features:
- Neural Network implemented with Torch
- Experience replay buffer
- Epsilon-greedy exploration
- Evolutionary hyperparameter optimization
- Modular architecture for easy extension

Proficient In:
    Classic control tasks (e.g., CartPole, Atari).
    Deterministic environments with fixed reward structures.

Best Used When:
    The environment is fully observable and stationary.
    You have large state/action spaces and can benefit from function approximation.
    You want to optimize performance over long-term training.
"""

import math
import torch
import random
import os
import copy
import numpy as np

from collections import deque

from src.agents.learning.utils.config_loader import load_global_config
from src.agents.learning.utils.neural_network import NeuralNetwork
from src.agents.learning.learning_memory import LearningMemory
from src.utils.buffer.replay_buffer import ReplayBuffer
from logs.logger import get_logger

logger = get_logger("Deep-Q Network Agent")

# ====================== Core DQN Agent ======================
class DQNAgent:
    """Standard DQN agent with neural network function approximation"""
    
    def __init__(self, agent_id, state_dim, action_dim):
        self.config = load_global_config()
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        #self.save_model = []

        # Get DQN-specific configuration
        dqn_config = self.config.get('dqn', {})
        self.hidden_dim = dqn_config.get('hidden_size', 128)
        self.gamma = dqn_config.get('gamma', 0.99)
        self.epsilon = dqn_config.get('epsilon', 1.0)
        self.epsilon_min = dqn_config.get('epsilon_min', 0.01)
        self.epsilon_decay = dqn_config.get('epsilon_decay', 0.995)
        self.lr = dqn_config.get('learning_rate', 0.001)
        self.batch_size = dqn_config.get('batch_size', 64)
        self.target_update = dqn_config.get('target_update_frequency', 100)

        self.learning_memory = LearningMemory()
        self.model_id = "DQN_Agent"

        # Initialize networks with validated dimensions
        self.policy_net = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim
        )
        self.target_net = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim
        )
        self.update_target_net()
        self._init_buffer()

        # Replay buffer
        logger.info("Deep-Q Network Agent has successfully initialized")

    def _init_buffer(self):
        # Use self.config to get buffer size
        buffer_size = self.config.get('dqn', {}).get('buffer_size', 10000)
        self.memory = ReplayBuffer(buffer_size)
        self.train_step = 0

    def update_target_net(self):
        """Hard update target network weights"""
        self.target_net.set_weights(self.policy_net.get_weights())

    def select_action(self, processed_state, explore=True):
        """Epsilon-greedy action selection"""
        if explore and torch.rand(1).item() < self.epsilon:
            return torch.randint(self.action_dim, size=())

        if not isinstance(processed_state, torch.Tensor):
            processed_state = torch.tensor(processed_state, dtype=torch.float32)
        
        processed_state = processed_state.float().unsqueeze(0)  # ensure batch dimension
        q_values = self.policy_net.forward(processed_state)
        return torch.argmax(q_values[0])

    def learn_step(self, experience_batch):
        """ standard DQN learning from batch """

    def store_transition(self, *transition):
        self.memory.push(transition)

    def evaluate(self, env, episodes=20, exploration_rate=0.05, visualize=False):
        """
        Comprehensive DQN evaluation with multiple performance metrics
        Args:
            env: Environment to evaluate in
            episodes: Number of evaluation episodes
            exploration_rate: Minimal exploration rate during evaluation
            visualize: Whether to render environment during evaluation
        
        Returns:
            Dict containing evaluation metrics
        """
        logger.info(f"Evaluating DQN Agent {self.agent_id} over {episodes} episodes")
        
        # Performance tracking
        total_rewards = []
        episode_lengths = []
        q_value_means = []
        q_value_stds = []
        action_distribution = {a: 0 for a in range(self.action_dim)}
        
        # Store original epsilon for restoration
        original_epsilon = self.epsilon
        self.epsilon = exploration_rate
        
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                if visualize:
                    env.render()
                    
                action = self.select_action(state, explore=True)
                action_int = action.item() if isinstance(action, torch.Tensor) else action
                next_state, reward, terminated, truncated, _ = env.step(action_int)
                done = terminated or truncated
                
                # Collect Q-value statistics
                state_tensor = torch.tensor([state], dtype=torch.float32)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                q_value_means.append(np.mean(q_values))
                q_value_stds.append(np.std(q_values))
                
                # Update trackers
                episode_reward += reward
                steps += 1
                action_distribution[action_int] += 1
                state = next_state

            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            logger.debug(f"Episode {ep+1}: Reward={episode_reward:.1f}, Steps={steps}")

        # Restore original exploration rate
        self.epsilon = original_epsilon
        
        # Calculate metrics
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        try:
            threshold = env.spec.reward_threshold
        except AttributeError:
            # Use 90% of max possible reward if threshold unavailable
            threshold = np.max(total_rewards) * 0.9 if total_rewards else 0
        success_rate = (np.array(total_rewards) >= threshold).mean()
        avg_length = np.mean(episode_lengths)
        
        # Normalize action distribution
        total_actions = sum(action_distribution.values())
        action_distribution = {k: v/total_actions for k, v in action_distribution.items()}
        
        # Q-value analysis
        avg_q_mean = np.mean(q_value_means)
        avg_q_std = np.mean(q_value_stds)
        
        # Target network divergence
        policy_weights_dict = self.policy_net.get_weights()
        target_weights_dict = self.target_net.get_weights()
        param_diff = []

        # Compare weight matrices for each layer
        for i in range(len(policy_weights_dict['Ws'])):
            pw = policy_weights_dict['Ws'][i].detach().cpu().numpy()
            tw = target_weights_dict['Ws'][i].detach().cpu().numpy()
            param_diff.append(np.linalg.norm(pw - tw))
            
        # Compare bias vectors for each layer
        for i in range(len(policy_weights_dict['bs'])):
            pb = policy_weights_dict['bs'][i].detach().cpu().numpy()
            tb = target_weights_dict['bs'][i].detach().cpu().numpy()
            param_diff.append(np.linalg.norm(pb - tb))

        avg_param_diff = np.mean(param_diff) if param_diff else 0.0
        
        return {
            'episodes': episodes,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'min_reward': min(total_rewards),
            'max_reward': max(total_rewards),
            'success_rate': success_rate,
            'avg_episode_length': avg_length,
            'action_distribution': action_distribution,
            'avg_q_value': avg_q_mean,
            'q_value_std': avg_q_std,
            'target_network_divergence': avg_param_diff,
            'replay_buffer_utilization': len(self.memory)/self.memory.capacity,
            'exploration_rate': exploration_rate,
            'detailed_rewards': total_rewards
        }

    def train(self):
        """Single training step from replay buffer"""
        if len(self.memory) < self.batch_size:
            return None # torch.tensor(50.0)
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to Torch arrays
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        current_q = self.policy_net.forward(states)
        next_q = self.target_net.forward(next_states)
        max_next_q = torch.max(next_q, dim=1).values
        
        target = current_q.clone().detach()
        batch_idx = torch.arange(self.batch_size)
        target[batch_idx, actions] = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Bellman equation update
        batch_idx = torch.arange(self.batch_size)
        target[batch_idx, actions] = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Backpropagation
        # self.policy_net.backward(states, target)
        loss = self.policy_net.train_step(states, target)
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        
        # Target network update
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.update_target_net()

        if loss is None:
            loss = torch.mean(torch.square(current_q - target))
        
        return loss.item() if isinstance(loss, torch.Tensor) else loss
        # return loss or torch.mean(torch.square(current_q - target))

    def save(self, path):
        """Save policy network weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        logger.info(f"Saved DQN model to {path}")


# ====================== Evolutionary Optimization ======================
class EvolutionaryTrainer:
    """Evolutionary hyperparameter optimization for DQN agents"""
    
    def __init__(self):
        base_config = load_global_config()
        evolutionary_config = base_config.get('evolutionary', {})

        self.pop_size = evolutionary_config.get('population_size', 10)
        self.generations = evolutionary_config.get('generations', 20)
        self.mutation_rate = evolutionary_config.get('mutation_rate', 0.2)
        self.evaluation_episodes = evolutionary_config.get('evaluation_episodes', 3)
        self.elite_ratio = evolutionary_config.get('elite_ratio', 0.3)
        logger.info(f"Evolutionary Trainer has succesfully initialized")

    def _random_config(self):
        """Generate random hyperparameter configuration under 'dqn' key"""
        return {
            'dqn': {
                'gamma': torch.clip(torch.normal(0.95, 0.02), 0.9, 0.999),
                'epsilon_decay': torch.clip(torch.normal(0.995, 0.002), 0.99, 0.999),
                'learning_rate': torch.clip(10**torch.uniform(-4, -2), 1e-4, 1e-2),
                'hidden_size': torch.choice([64, 128, 256]),
                'buffer_size': 10000,
                'batch_size': 64
            }
        }

    def _mutate(self, config):
        """Mutate parameters under 'dqn' key"""
        mutated = copy.deepcopy(config)
        dqn_config = mutated['dqn']  # Access nested parameters

        if torch.rand() < self.mutation_rate:
            dqn_config['gamma'] = torch.clip(dqn_config['gamma'] + torch.normal(0, 0.01), 0.9, 0.999)

        if torch.rand() < self.mutation_rate:
            dqn_config['learning_rate'] = torch.clip(
                dqn_config['learning_rate'] * torch.lognormal(0, 0.2), 1e-4, 1e-2)

        if torch.rand() < self.mutation_rate:
            dqn_config['epsilon_decay'] = torch.clip(
                dqn_config['epsilon_decay'] + torch.normal(0, 0.003), 0.99, 0.999)

        if torch.rand() < self.mutation_rate:
            dqn_config['hidden_size'] = torch.choice([64, 128, 256])

        return mutated

    def _evaluate(self, agent, episodes=3):
        """Evaluate agent's performance in environment"""
        total_rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, explore=False)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        return torch.mean(total_rewards)

    def evolve(self):
        """Run evolutionary training loop"""
        # Initialize population
        self.population = [DQNAgent(self.state_dim, self.action_dim, self._random_config())
                          for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = [(agent, self._evaluate(agent)) for agent in self.population]
            fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Selection (keep top 30%)
            elite = [agent for agent, _ in fitness[:int(0.3*self.pop_size)]]
            
            # Create next generation
            new_pop = []
            while len(new_pop) < self.pop_size:
                parent = torch.choice(elite)
                child_config = self._mutate(parent.config)
                new_pop.append(DQNAgent(self.state_dim, self.action_dim, child_config))
            
            self.population = new_pop
            print(f"Generation {gen+1} | Best Fitness: {fitness[0][1]:.1f}")
        
        return self.population[0]


# ====================== Unified Interface ======================
class UnifiedDQNAgent:
    """Unified interface for standard and evolutionary DQN"""
    
    def __init__(self, mode='standard', state_dim=None, action_dim=None, config=None, env=None, agent_id="Unified"):
        self.mode = mode
        self.config = config or {}
        self.env = env  # Store environment for all modes
        self.agent_id = agent_id
        
        if mode == 'standard':
            if state_dim is None or action_dim is None:
                raise ValueError("State and action dimensions required for standard mode")
            self.agent = DQNAgent(agent_id=self.agent_id, state_dim=state_dim, action_dim=action_dim)
        elif mode == 'evolutionary':
            if not env or not state_dim or not action_dim:
                raise ValueError("Evolutionary mode requires environment specs")
            self.trainer = EvolutionaryTrainer(env, state_dim, action_dim)
        else:
            raise ValueError("Invalid mode. Choose 'standard' or 'evolutionary'")

        logger.info(f"Unified Deep-Q Network Agent has succesfully initialized")

    def _run_validation(self, episodes=5):
        """Run evaluation episodes without exploration"""
        total_rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.select_action(state, explore=False)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        return torch.mean(total_rewards)

    def train(self, episodes=1000, validation_freq=50, validation_episodes=5,
              checkpoint_dir='checkpoints', early_stop_patience=20, target_reward=None):
        """Enhanced training loop with comprehensive features"""
        if self.mode == 'standard':
            if self.env is None:
                raise ValueError("Environment not provided for standard training")

            # Initialize training metrics
            episode_rewards = []
            episode_losses = []
            episode_lengths = []
            best_val_reward = -torch.inf
            early_stop_counter = 0
            os.makedirs(checkpoint_dir, exist_ok=True)

            for ep in range(episodes):
                state = self.env.reset()
                total_reward = 0
                steps = 0
                losses = []

                done = False
                while not done:
                    action = self.agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.store_transition(state, action, reward, next_state, done)
                    
                    loss = self.agent.train()
                    if loss is not None:
                        losses.append(loss)
                    
                    total_reward += reward
                    state = next_state
                    steps += 1

                # Episode statistics
                avg_loss = torch.mean(torch.tensor(losses)) if losses else 0
                episode_rewards.append(total_reward)
                episode_losses.append(avg_loss)
                episode_lengths.append(steps)

                # Calculate moving averages
                if len(episode_rewards) >= 10:
                    avg_reward_10 = torch.mean(torch.tensor(episode_rewards[-10:])).item()
                else:
                    avg_reward_10 = total_reward
                    
                if len(episode_losses) >= 10:
                    avg_loss_10 = torch.mean(torch.tensor(episode_losses[-10:])).item()
                else:
                    avg_loss_10 = avg_loss

                print(f"Episode {ep+1}/{episodes} | "
                      f"Reward: {total_reward:.1f} (Avg10: {avg_reward_10:.1f}) | "
                      f"Loss: {avg_loss:.4f} (Avg10: {avg_loss_10:.4f}) | "
                      f"Steps: {steps} | "
                      f"ε: {self.agent.epsilon:.3f}")

                # Validation and checkpointing
                if (ep + 1) % validation_freq == 0:
                    val_reward = self._run_validation(validation_episodes)
                    print(f"Validation | Avg Reward: {val_reward:.1f}")

                    if val_reward > best_val_reward:
                        best_val_reward = val_reward
                        self.save(os.path.join(checkpoint_dir, f'best_ep{ep+1}.npz'))
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1

                    # Early stopping
                    if early_stop_patience and early_stop_counter >= early_stop_patience:
                        print(f"Early stopping at episode {ep+1}")
                        break

                # Target reward termination
                if target_reward and avg_reward_10 >= target_reward:
                    print(f"Target reward achieved at episode {ep+1}!")
                    break

            print("Training completed")
            return {
                'rewards': episode_rewards,
                'losses': episode_losses,
                'lengths': episode_lengths
            }
        else:
            self.agent = self.trainer.evolve()

    def save(self, path):
        """Save policy network weights"""
        weights = self.agent.policy_net.get_weights()
        torch.savez(path, *weights)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load policy network weights"""
        with torch.load(path) as data:
            weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        self.agent.policy_net.set_weights(weights)
        print(f"Model loaded from {path}")

    def act(self, state, explore=False):
        return self.agent.select_action(state, explore)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Deep-Q Network Agent ===\n")

    config = load_global_config()
    agent_id = None

    class MockEnv:
        def reset(self):
            return torch.randn(4)
        
        def step(self, action):
            return torch.randn(4), torch.rand(), random.choice([True, False]), {}

    agent1 = DQNAgent(
        state_dim=4, 
        action_dim=2, 
        agent_id=agent_id
    )
    
    agent2 = EvolutionaryTrainer()
    print(f"\n{agent1}\n{agent2}\n")
    print("\n=== Successfully Ran Deep-Q Network Agent ===\n")

if __name__ == "__main__":
    print("\n * * * * Phase 2 * * * *\n=== Running Deep-Q Network Agent ===\n")

    # Load config
    config = load_global_config()

    # Mock environment (replace with Gym-style env for real use)
    class MockEnv:
        def reset(self):
            return torch.randn(4)  # Example state vector

        def step(self, action):
            next_state = torch.randn(4)
            reward = torch.rand(1).item()
            done = random.random() < 0.1
            return next_state, reward, done, {}

    env = MockEnv()
    state_dim = 4
    action_dim = 2

    # --- 1. Directly using DQNAgent ---
    agent1 = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        agent_id=agent_id
    )
    print(f"Initialized DQNAgent: {agent1.model_id}")

    # Optionally run a single episode
    state = env.reset()
    done = False
    while not done:
        action = agent1.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent1.store_transition(state, action, reward, next_state, done)
        agent1.train()
        state = next_state

    # --- 2. Evolutionary training interface ---
    trainer = EvolutionaryTrainer()
    # Optional: evolve population
    # best_agent = trainer.evolve()

    # --- 3. Unified agent with training ---
    unified_agent = UnifiedDQNAgent(
        mode='standard',
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        env=env
    )
    # Optional: Run full training loop
    training_metrics = unified_agent.train(
        episodes=50,
        validation_freq=10,
        validation_episodes=2,
        checkpoint_dir="checkpoints",
        early_stop_patience=5,
        target_reward=10
    )

    print("\n=== All Agents Initialized and Tested ===\n")
