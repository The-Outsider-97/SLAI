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
- Neural Network implemented with NumPy
- Experience replay buffer
- Epsilon-greedy exploration
- Evolutionary hyperparameter optimization
- Modular architecture for easy extension
"""

import os, sys
import torch
import random
from collections import deque
import copy

from src.collaborative.shared_memory import SharedMemory

# ====================== Neural Network Core ======================
class NeuralNetwork:
    """Simple 3-layer neural network with manual backpropagation"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        # He initialization with ReLU
        self.W1 = torch.randn(input_dim, hidden_dim) * torch.sqrt(2./input_dim)
        self.b1 = torch.zeros(hidden_dim)
        self.W2 = torch.randn(hidden_dim, hidden_dim) * torch.sqrt(2./hidden_dim)
        self.b2 = torch.zeros(hidden_dim)
        self.W3 = torch.randn(hidden_dim, output_dim) * torch.sqrt(2./hidden_dim)
        self.b3 = torch.zeros(output_dim)

        # Intermediate values for backprop
        self._cache = {}

    def forward(self, X):
        """Forward pass with ReLU activation"""
        self._cache['z1'] = X @ self.W1 + self.b1
        self._cache['a1'] = torch.maximum(0, self._cache['z1'])  # ReLU
        self._cache['z2'] = self._cache['a1'] @ self.W2 + self.b2
        self._cache['a2'] = torch.maximum(0, self._cache['z2'])
        self._cache['out'] = self._cache['a2'] @ self.W3 + self.b3
        return self._cache['out']

    def backward(self, X, y, learning_rate):
        """Manual backpropagation with MSE loss"""
        m = X.shape[0]  # Batch size
        out = self._cache['out']
        
        # Output layer gradient
        dout = (out - y) * 2/m
        dW3 = self._cache['a2'].T @ dout
        db3 = torch.sum(dout, axis=0)
        
        # Hidden layer 2 gradient
        da2 = dout @ self.W3.T
        dz2 = da2 * (self._cache['a2'] > 0)
        dW2 = self._cache['a1'].T @ dz2
        db2 = torch.sum(dz2, axis=0)
        
        # Hidden layer 1 gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self._cache['a1'] > 0)
        dW1 = X.T @ dz1
        db1 = torch.sum(dz1, axis=0)
        
        # Parameter updates
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights


# ====================== Experience Replay Buffer ======================
class ReplayBuffer:
    """Experience replay buffer with uniform sampling"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        """Store transition (state, action, reward, next_state, done)"""
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """Random batch of transitions"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


# ====================== Core DQN Agent ======================
class DQNAgent:
    """Standard DQN agent with neural network function approximation"""
    
    def __init__(self, state_dim, action_dim, config):
        # Network parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = config.get('hidden_size', 128)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.lr = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.target_update = config.get('target_update_frequency', 100)

        self.shared_memory = SharedMemory
        self.config = config or {}
        self.model_id = "DQN_Agent"
        
        # Networks
        self.policy_net = NeuralNetwork(state_dim, action_dim, self.hidden_dim)
        self.target_net = NeuralNetwork(state_dim, action_dim, self.hidden_dim)
        self.update_target_net()
        
        # Replay buffer
        self.memory = ReplayBuffer(config.get('buffer_size', 10000))
        self.train_step = 0

    def update_target_net(self):
        """Hard update target network weights"""
        self.target_net.set_weights(self.policy_net.get_weights())

    def select_action(self, state, explore=True):
        """Epsilon-greedy action selection"""
        if explore and torch.rand() < self.epsilon:
            return torch.randint(self.action_dim)
        
        q_values = self.policy_net.forward(torch.Tensor([state]))
        return torch.argmax(q_values[0])

    def store_transition(self, *transition):
        self.memory.push(transition)

    def train(self):
        """Single training step from replay buffer"""
        if len(self.memory) < self.batch_size:
            return 50
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to NumPy arrays
        states = torch.Tensor(states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        dones = torch.Tensor(dones)
        
        # Calculate target Q-values
        current_q = self.policy_net.forward(states)
        next_q = self.target_net.forward(next_states)
        max_next_q = torch.max(next_q, axis=1)
        target = current_q.copy()
        
        # Bellman equation update
        batch_idx = torch.arange(self.batch_size)
        target[batch_idx, actions] = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Backpropagation
        self.policy_net.backward(states, target, self.lr)
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        
        # Target network update
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.update_target_net()
        
        return torch.mean(torch.square(current_q - target))


# ====================== Evolutionary Optimization ======================
class EvolutionaryTrainer:
    """Evolutionary hyperparameter optimization for DQN agents"""
    
    def __init__(self, env, state_dim, action_dim, 
                 population_size=10, generations=20, mutation_rate=0.2):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []

    def _random_config(self):
        """Generate random hyperparameter configuration"""
        return {
            'gamma': torch.clip(torch.normal(0.95, 0.02), 0.9, 0.999),
            'epsilon_decay': torch.clip(torch.normal(0.995, 0.002), 0.99, 0.999),
            'learning_rate': torch.clip(10**torch.uniform(-4, -2), 1e-4, 1e-2),
            'hidden_size': torch.choice([64, 128, 256]),
            'buffer_size': 10000,
            'batch_size': 64
        }

    def _mutate(self, config):
        """Apply Gaussian mutation to hyperparameters"""
        mutated = copy.deepcopy(config)
        
        if torch.rand() < self.mutation_rate:
            mutated['gamma'] = torch.clip(config['gamma'] + torch.normal(0, 0.01), 0.9, 0.999)
        
        if torch.rand() < self.mutation_rate:
            mutated['learning_rate'] = torch.clip(
                config['learning_rate'] * torch.lognormal(0, 0.2), 1e-4, 1e-2)
        
        if torch.rand() < self.mutation_rate:
            mutated['epsilon_decay'] = torch.clip(
                config['epsilon_decay'] + torch.normal(0, 0.003), 0.99, 0.999)
        
        if torch.rand() < self.mutation_rate:
            mutated['hidden_size'] = torch.choice([64, 128, 256])
        
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
    
    def __init__(self, mode='standard', state_dim=None, action_dim=None, config=None, env=None):
        self.mode = mode
        self.config = config or {}
        self.env = env  # Store environment for all modes
        
        if mode == 'standard':
            if state_dim is None or action_dim is None:
                raise ValueError("State and action dimensions required for standard mode")
            self.agent = DQNAgent(state_dim, action_dim, self.config)
        elif mode == 'evolutionary':
            if not env or not state_dim or not action_dim:
                raise ValueError("Evolutionary mode requires environment specs")
            self.trainer = EvolutionaryTrainer(env, state_dim, action_dim)
        else:
            raise ValueError("Invalid mode. Choose 'standard' or 'evolutionary'")

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
                avg_loss = torch.mean(losses) if losses else 0
                episode_rewards.append(total_reward)
                episode_losses.append(avg_loss)
                episode_lengths.append(steps)

                # Calculate moving averages
                avg_reward_10 = torch.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
                avg_loss_10 = torch.mean(episode_losses[-10:]) if len(episode_losses) >= 10 else avg_loss

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
    # Example environment setup
    class MockEnv:
        def reset(self):
            return torch.randn(4)
        
        def step(self, action):
            return torch.randn(4), torch.rand(), random.choice([True, False]), {}
    
    # Initialize agent
    agent = UnifiedDQNAgent(mode='standard',
                        state_dim=4,
                        action_dim=2,
                        config={'hidden_size': 128, 'learning_rate': 0.001},
                        env=MockEnv())

    # Start training with enhanced parameters
    metrics = agent.train(
        episodes=500,
        validation_freq=20,
        validation_episodes=3,
        checkpoint_dir='dqn_checkpoints',
        early_stop_patience=5,
        target_reward=195  # Example for CartPole target
    )
