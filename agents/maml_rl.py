import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward'])

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class MAMLAgent:
    def __init__(self, state_size, action_size, hidden_size=64, meta_lr=0.001, inner_lr=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.policy = PolicyNetwork(state_size, action_size, hidden_size)
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr

    def get_action(self, state, policy=None):
        if policy is None:
            policy = self.policy
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def collect_trajectory(self, env, policy=None, max_steps=200):
        trajectory = []
        state, _ = env.reset()
        for _ in range(max_steps):
            action, log_prob = self.get_action(state, policy)
            next_state, reward, done, truncated, _ = env.step(action)

            transition = Transition(state, action, reward)
            trajectory.append(transition)
            state = next_state
            if done or truncated:
                break
        return trajectory

    def compute_loss(self, trajectory, policy):
        rewards = [t.reward for t in trajectory]
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_rewards.insert(0, G)

        log_probs = []
        for t in trajectory:
            _, log_prob = self.get_action(t.state, policy)
            log_probs.append(log_prob)

        loss = []
        for log_prob, G in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * G)
        return torch.stack(loss).sum()

    def inner_update(self, env):
        trajectory = self.collect_trajectory(env)
        loss = self.compute_loss(trajectory, self.policy)

        grads = torch.autograd.grad(loss, self.policy.parameters(), create_graph=True)
        adapted_policy = PolicyNetwork(self.state_size, self.action_size)

        # Apply gradient step manually
        for (name, param), grad in zip(self.policy.named_parameters(), grads):
            adapted_param = param - self.inner_lr * grad
            getattr(adapted_policy, name.replace('.', '_')).data = adapted_param.data.clone()

        return adapted_policy

    def meta_update(self, tasks, inner_steps=1):
        meta_loss = 0.0

        for env, _ in tasks:
            # Inner loop: adapt policy to the task
            adapted_policy = self.inner_update(env)

            # Collect trajectory with adapted policy
            trajectory = self.collect_trajectory(env, policy=adapted_policy)
            task_loss = self.compute_loss(trajectory, adapted_policy)
            meta_loss += task_loss

        meta_loss /= len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
