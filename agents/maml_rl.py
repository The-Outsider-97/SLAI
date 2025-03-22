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

        pass

    def clone_policy(self, policy):
        import copy
        cloned_policy = copy.deepcopy(policy)
        return cloned_policy

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

    def inner_update(self, env, log=False):
        """
        Performs an inner loop adaptation step for MAML.
        Args:
            env (gym.Env): Environment to collect data from.
            log (bool): Whether to print debug logs.
        Returns:
            adapted_policy (nn.Module): The policy network after one inner update.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Ensure policy is on the correct device
        policy = self.policy.to(device)

        # Collect trajectory data from the environment
        trajectories = self.collect_trajectory(env, policy)

        # Compute the inner loop loss on the collected trajectory
        loss = self.compute_loss(trajectories, policy)

        # Compute gradients w.r.t. policy parameters
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=policy.parameters(),
            create_graph=True,
            retain_graph=True,
            allow_unused=True  # Safe for some layers (e.g., Dropout)
        )

        # Clone the current policy for adaptation
        adapted_policy = self.clone_policy(policy)

        # Prepare the adapted parameters (manual SGD step)
        adapted_params = {}

        for (name, param), grad in zip(policy.named_parameters(), grads):
            if grad is None:
                if log:
                    print(f"[WARNING] No grad for {name}, skipping...")
                adapted_params[name] = param.detach().clone()
                continue

            # Gradient descent step: theta' = theta - alpha * grad
            adapted_param = param - self.inner_lr * grad
            adapted_params[name] = adapted_param.detach().clone()  # Detach for safety

            if log:
                print(f"[Inner Update] Param: {name}, Grad Norm: {grad.norm():.4f}")

        # Load the adapted parameters into the cloned policy
        adapted_state_dict = adapted_policy.state_dict()
        for name in adapted_state_dict.keys():
            adapted_state_dict[name] = adapted_params[name]

        adapted_policy.load_state_dict(adapted_state_dict)

        if log:
            print("[Inner Update] Completed parameter adaptation.")

        return adapted_policy.to(device)

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
