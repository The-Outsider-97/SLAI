import os, sys
import yaml
import gymnasium as gym
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from logs.logger import get_logger
from agents.dqn_agent import DQNAgent
from utils.logger import setup_logger
from agents.evolutionary_dqn import EvolutionaryTrainer

state_dim = 4
action_dim = 2

config = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.990,
    'batch_size': 128,
    'learning_rate': 0.002,
    'memory_size': 10000,
    'hidden_size': 128
}

agent = DQNAgent(state_size=state_dim, action_size=action_dim, config=config)

# Training from dataset
agent.pretrain_from_dataset('datasets/sample_replay.json', epochs=20)

logger = setup_logger('SLAI-CartPole', level=logging.DEBUG)

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    trainer = EvolutionaryTrainer(env, population_size=10, generations=20)
    best_agent = trainer.evolve(state_size, action_size)

    config = {
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.990,
        'batch_size': 128,
        'learning_rate': 0.002,
        'memory_size': 10000,
        'hidden_size': 128
    }

    agent = DQNAgent(state_size, action_size, config)
    episodes = 500
    target_update_frequency = 100

    rewards = []  # <---- Add this to track episode rewards

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Warm-up condition BEFORE calling train_step()
            if len(agent.memory) >= agent.batch_size:
                    agent.train_step()

            if done or truncated:
                logger.info(f"Episode {e+1}/{episodes} - Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        rewards.append(total_reward)  # <---- Store episode reward here

        if e % target_update_frequency == 0:
            agent.update_target_network()

    env.close()

    # Plot rewards after training
    plt.plot(rewards)
    plt.title('Reward Trend')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/reward_trend.png')
    plt.show()  # Optional, if you want to display it directly

if __name__ == "__main__":
    main()
