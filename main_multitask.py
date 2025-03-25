import time
import sys
import torch
import logging
import torch.optim as optim
import torch.nn.functional as F
from logger import get_logger
from tasks.task_sampler import TaskSampler
from agents.multitask_rl import MultiTaskPolicy
from utils.logger import setup_logger
# ===============================
# Initialize Logger
# ===============================
logger = setup_logger('SLAI-MultiTask', level=logging.DEBUG)


# ===============================
# Train MultiTask Policy Function
# ===============================
def train_multitask_policy(config=None):
    # ===============================
    # Default Configuration
    # ===============================
    if config is None:
        config = {
            'tasks': ['CartPole-v1', 'CartPole-v1', 'CartPole-v1'],  # Same task with variations recommended!
            'num_tasks': 3,
            'task_embedding_size': 16,
            'hidden_size': 128,
            'lr': 0.001,
            'epochs': 500,
            'max_timesteps': 500,
            'gamma': 0.99
        }

    logger.info(f"Starting Multi-Task RL Training with config: {config}")

    # ===============================
    # Task Sampler Initialization
    # ===============================
    sampler = TaskSampler(
        base_task=config['tasks'][0],  # Base task to create variations
        num_tasks=config['num_tasks'],
        seed=42
    )

    # Sample tasks and environments
    envs = [sampler.sample_task(return_params=True) for _ in range(config['num_tasks'])]

    logger.info(f"Sampled {len(envs)} environments for multi-task training.")

    # ===============================
    # Setup State/Action Dimensions
    # ===============================
    env, task_params = envs[0]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    logger.info(f"State Size: {state_size}, Action Size: {action_size}")

    # ===============================
    # Initialize MultiTask Policy & Optimizer
    # ===============================
    multitask_policy = MultiTaskPolicy(
        state_size=state_size,
        action_size=action_size,
        task_embedding_size=config['task_embedding_size'],
        num_tasks=config['num_tasks'],
        hidden_size=config['hidden_size']
    )

    optimizer = optim.Adam(multitask_policy.parameters(), lr=config['lr'])

    logger.info(
        f"Initialized MultiTaskPolicy with:\n"
        f"  - State Size: {state_size}\n"
        f"  - Action Size: {action_size}\n"
        f"  - Task Embedding Size: {config['task_embedding_size']}\n"
        f"  - Num Tasks: {config['num_tasks']}\n"
        f"  - Hidden Size: {config['hidden_size']}\n"
        f"Initialized Optimizer with Learning Rate: {config['lr']}"
    )

    # ===============================
    # Training Loop
    # ===============================
    for epoch in range(1, config['epochs'] + 1):
        epoch_rewards = []

        for task_id, (env, params) in enumerate(envs):
            result = env.reset()

            if isinstance(result, tuple):
                state, _ = result
            else:
                state = result

            total_reward = 0
            log_probs = []
            rewards = []

            for timestep in range(config['max_timesteps']):
                # Prepare input tensors
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                task_id_tensor = torch.LongTensor([task_id])

                # Forward pass through multitask policy
                probs = multitask_policy(state_tensor, task_id_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                log_prob = dist.log_prob(action)
                next_state, reward, done, truncated, _ = env.step(action.item())

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            # Compute REINFORCE Loss
            returns = compute_returns(rewards, config['gamma'])
            returns = torch.FloatTensor(returns)

            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = torch.stack(policy_loss).sum()

            # Optimize policy
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            logger.info(f"Epoch {epoch}/{config['epochs']} | Task {task_id} | Reward: {total_reward:.2f}")
            epoch_rewards.append(total_reward)

        # Epoch summary
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        logger.info(f"Epoch {epoch} | Average Reward Across Tasks: {avg_reward:.2f}")

    logger.info("Multi-Task RL Training Completed.")


# ===============================
# Compute Discounted Returns
# ===============================
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# ===============================
# Main Function Entry Point
# ===============================
if __name__ == "__main__":
    multitask_config = {
        'tasks': ['CartPole-v1'],  # Recommend param variations via TaskSampler!
        'num_tasks': 3,
        'task_embedding_size': 16,
        'hidden_size': 128,
        'lr': 0.001,
        'epochs': 500,
        'max_timesteps': 500,
        'gamma': 0.99
    }

    train_multitask_policy(config=multitask_config)
