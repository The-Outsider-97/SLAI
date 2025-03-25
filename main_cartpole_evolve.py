import os
import sys
import time
import gymnasium as gym
import torch
import logging
from logger import get_logger
from agents.evolutionary_dqn import EvolutionaryTrainer
from utils.logger import setup_logger

# ===============================
# Initialize Logger
# ===============================
logger = setup_logger('SLAI-CartPole-Evolution', level=logging.DEBUG)


# ===============================
# Train with Evolutionary DQN Function
# ===============================
def train_evolutionary_dqn(config=None):
    # ===============================
    # Default Configuration
    # ===============================
    if config is None:
        config = {
            'env_name': 'CartPole-v1',
            'population_size': 10,
            'generations': 10,
            'mutation_rate': 0.2,
            'episodes_per_evaluation': 5,
            'state_size': None,  # Auto-fill later
            'action_size': None,  # Auto-fill later
            'max_timesteps': 500,
            'save_best_agent': True,
            'best_model_path': 'logs/best_evolved_dqn_agent.pth'
        }

    logger.info(f"Starting Evolutionary DQN Training with config: {config}")

    # ===============================
    # Initialize Gym Environment
    # ===============================
    env = gym.make(config['env_name'])
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    config['state_size'] = state_size
    config['action_size'] = action_size

    logger.info(f"Environment: {config['env_name']}, State size: {state_size}, Action size: {action_size}")

    # ===============================
    # Initialize Evolutionary Trainer
    # ===============================
    evolutionary_trainer = EvolutionaryTrainer(
        env=env,
        population_size=config['population_size'],
        generations=config['generations'],
        mutation_rate=config['mutation_rate']
    )

    logger.info("Initialized EvolutionaryTrainer.")

    # ===============================
    # Start Evolution Process
    # ===============================
    start_time = time.time()

    best_agent = evolutionary_trainer.evolve(
        state_size=state_size,
        action_size=action_size
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Evolution completed in {elapsed_time:.2f} seconds.")

    # ===============================
    # Save the Best Agent
    # ===============================
    if config['save_best_agent']:
        os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
        torch.save(best_agent.policy_net.state_dict(), config['best_model_path'])
        logger.info(f"Saved best evolved agent to {config['best_model_path']}")

    # ===============================
    # Evaluate Best Agent Performance
    # ===============================
    final_score = evolutionary_trainer.evaluate_agent(
        best_agent,
        episodes=config['episodes_per_evaluation']
    )
    logger.info(f"Best Evolved Agent Average Score (over {config['episodes_per_evaluation']} episodes): {final_score:.2f}")

    env.close()


# ===============================
# Main Function Entry Point
# ===============================
if __name__ == "__main__":
    evolutionary_config = {
        'env_name': 'CartPole-v1',
        'population_size': 10,
        'generations': 10,
        'mutation_rate': 0.2,
        'episodes_per_evaluation': 5,
        'save_best_agent': True,
        'best_model_path': 'logs/best_evolved_dqn_agent.pth'
    }

    train_evolutionary_dqn(config=evolutionary_config)
