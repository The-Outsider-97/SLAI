import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RLAgent:
    """
    A simplified RL agent for demonstration purposes.
    Supports initialization with hyperparameters, training over episodes,
    and evaluation of cumulative reward.
    """

    def __init__(self, learning_rate=0.01, num_layers=2, activation_function='relu'):
        """
        Initialize the RL agent with specific hyperparameters.
        
        Args:
            learning_rate (float): Learning rate for the policy.
            num_layers (int): Number of layers in the policy network.
            activation_function (str): Activation function used in the network.
        """
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.activation_function = activation_function

        logger.info(
            f"Initialized RLAgent with learning_rate={self.learning_rate}, "
            f"num_layers={self.num_layers}, activation_function={self.activation_function}"
        )

        self.policy = self._build_policy()

    def execute(self, task_data):
        """
        Execute the RSI task using given data. Required for collaboration system.
        """
        print("[RSI_Agent] Executing task:", task_data)

        # Run training with dynamic self-tuning
        self.train()

        # Collect metrics
        evaluation = self.evaluate()

        # Optionally write to shared memory
        self.shared_memory.set("rsi_agent_last_eval", evaluation)

        return evaluation
        
    def _build_policy(self):
        """
        Constructs a simple policy model.
        This is a placeholder for an actual neural network or policy algorithm.
        
        Returns:
            dict: Simulated policy parameters.
        """
        policy = {
            'weights': np.random.randn(self.num_layers, 10),
            'activation': self.activation_function
        }
        logger.info("Policy model built with weights shape: %s", policy['weights'].shape)
        return policy

    def train(self, episodes=100):
        """
        Trains the RL agent over a number of episodes.
        Simulates policy updates.
        
        Args:
            episodes (int): Number of training episodes.
        """
        logger.info("Training started for %d episodes...", episodes)

        for episode in range(episodes):
            reward = self.run_episode(train_mode=True)
            logger.debug("Episode %d reward: %.2f", episode + 1, reward)

        logger.info("Training completed after %d episodes.", episodes)

    def evaluate(self, eval_episodes=10):
        """
        Evaluates the RL agent over a number of episodes.
        
        Args:
            eval_episodes (int): Number of evaluation episodes.
        
        Returns:
            float: Average cumulative reward.
        """
        logger.info("Evaluating agent over %d episodes...", eval_episodes)

        rewards = []
        for _ in range(eval_episodes):
            reward = self.run_episode(train_mode=False)
            rewards.append(reward)

        avg_reward = np.mean(rewards)
        logger.info("Evaluation complete. Average reward: %.2f", avg_reward)
        return avg_reward

    def run_episode(self, train_mode=False):
        """
        Runs a single episode of interaction with the environment.
        
        Args:
            train_mode (bool): Whether the agent is training or evaluating.
        
        Returns:
            float: Cumulative reward for the episode.
        """
        steps = 100  # Fixed number of steps per episode
        cumulative_reward = 0

        for step in range(steps):
            action = self._select_action()
            reward = self._get_reward(action)
            cumulative_reward += reward

            if train_mode:
                self._update_policy(action, reward)

        return cumulative_reward

    def _select_action(self):
        """
        Selects an action based on the policy.
        This example uses random actions for simplicity.
        
        Returns:
            int: Selected action.
        """
        action = random.choice([0, 1])  # e.g., binary action space
        logger.debug("Action selected: %d", action)
        return action

    def _get_reward(self, action):
        """
        Returns a reward based on the action.
        Simulates an environment response.
        
        Args:
            action (int): The action taken.
        
        Returns:
            float: Simulated reward.
        """
        if action == 1:
            reward = np.random.normal(1.0, 0.1)
        else:
            reward = np.random.normal(0.5, 0.1)

        logger.debug("Reward received: %.2f", reward)
        return reward

    def _update_policy(self, action, reward):
        """
        Updates the policy based on the action and reward.
        Simulates weight adjustment.
        
        Args:
            action (int): Action taken.
            reward (float): Reward received.
        """
        learning_factor = self.learning_rate * reward
        noise = np.random.randn(*self.policy['weights'].shape)
        self.policy['weights'] += learning_factor * noise

        logger.debug("Policy updated with learning factor: %.4f", learning_factor)

if __name__ == "__main__":
    # Example usage
    agent = RLAgent(learning_rate=0.01, num_layers=3, activation_function='relu')
    agent.train(episodes=50)
    avg_reward = agent.evaluate(eval_episodes=5)
    print("\nAverage reward after evaluation:", avg_reward)

def execute(self, task_data):
    train_mode = task_data.get("train_mode", False)
    if train_mode:
        self.train(episodes=10)
        return {"status": "trained", "agent": "RLAgent"}
    else:
        avg_reward = self.evaluate()
        return {"status": "evaluated", "avg_reward": avg_reward}
