import os, sys
import logging
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

    def clone_policy(self, policy):
        import copy
        return copy.deepcopy(policy)

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
            trajectory.append(Transition(state, action, reward))
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

        log_probs = [self.get_action(t.state, policy)[1] for t in trajectory]
        loss = [-lp * G for lp, G in zip(log_probs, discounted_rewards)]
        return torch.stack(loss).sum()

    def inner_update(self, env, log=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = self.policy.to(device)

        trajectory = self.collect_trajectory(env, policy)
        loss = self.compute_loss(trajectory, policy)

        grads = torch.autograd.grad(loss, policy.parameters(), create_graph=True, retain_graph=True, allow_unused=True)
        adapted_policy = self.clone_policy(policy)

        adapted_params = {
            name: (param - self.inner_lr * grad if grad is not None else param.detach().clone())
            for (name, param), grad in zip(policy.named_parameters(), grads)
        }

        adapted_state_dict = adapted_policy.state_dict()
        for name in adapted_state_dict:
            adapted_state_dict[name] = adapted_params[name]
        adapted_policy.load_state_dict(adapted_state_dict)

        return adapted_policy.to(device)

    def meta_update(self, tasks, inner_steps=1):
        meta_loss = 0.0
        for env, _ in tasks:
            adapted_policy = self.inner_update(env)
            trajectory = self.collect_trajectory(env, adapted_policy)
            task_loss = self.compute_loss(trajectory, adapted_policy)
            meta_loss += task_loss

        meta_loss /= len(tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.meta_optimizer.step()
        return meta_loss.item()

    def execute(self, task_data):
        tasks = task_data.get("tasks", [])
        loss = self.meta_update(tasks)
        return {"status": "success", "agent": "MAMLAgent", "meta_loss": loss}

    def train(self, num_tasks=3, episodes_per_task=5):
        import gym
        print("[MAMLAgent] Starting mock meta-training loop...")
        self.mock_rewards = []
        for t in range(num_tasks):
            task_rewards = []
            for e in range(episodes_per_task):
                reward = 80 + np.random.rand() * 10
                task_rewards.append(reward)
            avg = np.mean(task_rewards)
            self.mock_rewards.append(avg)
        print("[MAMLAgent] Meta-training complete.")

    def evaluate(self):
        print("[MAMLAgent] Running mock evaluation...")
        rewards = self.mock_rewards if hasattr(self, "mock_rewards") else [80 + i for i in range(10)]
        return {
            "average_reward": float(np.mean(rewards)),
            "reward_trace": rewards
        }
