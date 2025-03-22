import gymnasium as gym
import numpy as np
import torch
import logging
from agents.dqn_agent import DQNAgent
from utils.logger import setup_logger
from agents.evolutionary_dqn import EvolutionaryTrainer

# env = gym.make('CartPole-v1')
# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n

# trainer = EvolutionaryTrainer(env, population_size=10, generations=20)
# best_agent = trainer.evolve(state_size, action_size)

logger = setup_logger('SLAI-CartPole', level=logging.DEBUG)

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Evolutionary trainer (optional pre-evolution before RL training)
    trainer = EvolutionaryTrainer(env, population_size=10, generations=20)
    best_agent = trainer.evolve(state_size, action_size)

    config = {
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'learning_rate': 0.001,
        'memory_size': 10000
        'hidden_size': 128
    }

    agent = DQNAgent(state_size, action_size, config)
    episodes = 500
    target_update_frequency = 10

    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

            if done or truncated:
                logger.info(f"Episode {e+1}/{episodes} - Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if e % target_update_frequency == 0:
            agent.update_target_network()

    env.close()

if __name__ == "__main__":
    main()
