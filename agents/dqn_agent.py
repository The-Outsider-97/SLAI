import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging
from utils.data_loader import FlexibleDataLoader as DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ReplayBuffer:
    """
    Experience Replay Buffer to store agent transitions.
    Supports uniform sampling. Prioritized experience replay can be added later.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        logger.info(f"ReplayBuffer initialized with capacity {self.capacity}.")

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Randomly sample a batch from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    A Deep Q-Network (DQN) implemented in PyTorch.
    Supports arbitrary hidden sizes and activation functions.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, activation=nn.ReLU):
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, action_dim)
        )
        logger.info(f"DQNNetwork initialized with state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")

    def forward(self, state):
        return self.layers(state)


class DQNAgent:
    """
    A modular Deep Q-Learning Agent with experience replay and target networks.
    Supports pretraining with datasets and online learning.
    """

    def __init__(self, state_size, action_size, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.hidden_size = config.get('hidden_size', 128)
        self.target_update_frequency = config.get('target_update_frequency', 10)
        self.batch_size = config.get('batch_size', 64)
        self.memory_capacity = config.get('memory_capacity', 10000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DQNAgent initialized on {self.device}.")

        # Networks
        self.policy_net = DQNNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Experience Replay
        self.memory = ReplayBuffer(capacity=self.memory_capacity)

        # Synchronize target network
        self.update_target_network(hard_update=True)

        # Step counter for periodic updates
        self.training_steps = 0

    def update_target_network(self, hard_update=False):
        """Soft or hard update for target network weights."""
        if hard_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug("Hard update performed on target network.")
        else:
            tau = 0.005
            for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, explore=True):
        """
        Epsilon-greedy action selection.
        If explore is False, always exploit.
        """
        if explore and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            logger.debug(f"Random action {action} selected under epsilon {self.epsilon}")
            return action

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = q_values.argmax().item()
        logger.debug(f"Greedy action {action} selected.")
        return action

    def act(self, state):
        """
        Alias for select_action without exploration.
        Used in evaluation (EvolutionaryTrainer) and main_cartpole.py.
        """
        return self.select_action(state, explore=False)

    def act(self, state):
        """Alias for select_action without exploration (for evaluation purposes)."""
        return self.select_action(state, explore=False)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        logger.debug(f"Transition stored. Memory size: {len(self.memory)}")

    def train_step(self):
        """One step of training from replay buffer."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q estimates
        q_values = self.policy_net(states).gather(1, actions)

        # Compute targets
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss calculation
        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.debug(f"Train step completed. Loss: {loss.item()}")

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodic target net update
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network(hard_update=True)

        return loss.item()

    def pretrain_from_dataset(self, dataset_path: str, epochs: int = 10):
        """
        Load offline dataset and pretrain policy network.
        Dataset must include keys: states, actions, rewards, next_states, dones.
        """
        logger.info(f"Starting pretraining from dataset {dataset_path}")
        loader = DataLoader(validation_schema={
            'states': list,
            'actions': list,
            'rewards': list,
            'next_states': list,
            'dones': list
        })

        dataset = loader.load(dataset_path)
        num_samples = len(dataset['states'])

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            total_loss = 0.0
            batches = 0

            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                batch = {
                    key: [dataset[key][i] for i in batch_indices]
                    for key in dataset
                }

                states = torch.FloatTensor(batch['states']).to(self.device)
                actions = torch.LongTensor(batch['actions']).unsqueeze(1).to(self.device)
                rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
                next_states = torch.FloatTensor(batch['next_states']).to(self.device)
                dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)

                q_values = self.policy_net(states).gather(1, actions)

                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + (1 - dones) * self.gamma * next_q_values

                loss = nn.MSELoss()(q_values, target_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / batches
            logger.info(f"Epoch [{epoch + 1}/{epochs}] - Avg Loss: {avg_loss:.4f}")
