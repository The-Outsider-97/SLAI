import argparse
import logging
import json
import torch
import numpy as np
import itertools
from hyperparam_tuning.tuner import HyperParamTuner
from agent.rl_agent import RLAgent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Real RL agent evaluation function
def rl_agent_evaluation(params):
    """
    Evaluation function for the RL agent.
    Hyperparameters are passed from the tuner (Grid/Bayesian).

    Args:
        params (dict): Dictionary of hyperparameters to apply to the agent.

    Returns:
        float: Average reward across evaluation episodes.
    """
    learning_rate = params.get('learning_rate', 0.01)
    num_layers = params.get('num_layers', 2)
    activation_function = params.get('activation', 'relu')

    logger.info("Initializing RLAgent with hyperparameters: %s", params)

    agent = RLAgent(
        learning_rate=learning_rate,
        num_layers=num_layers,
        activation_function=activation_function
    )

    agent.train(episodes=100)
    avg_reward = agent.evaluate(eval_episodes=10)

    logger.info("Evaluation complete. Average reward: %.4f", avg_reward)
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning with RLAgent")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to hyperparameter config JSON file.'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default='bayesian',
        choices=['bayesian', 'grid'],
        help='Choose optimization strategy: bayesian or grid.'
    )

    parser.add_argument(
        '--n_calls',
        type=int,
        default=20,
        help='(Bayesian only) Number of optimization calls.'
    )

    parser.add_argument(
        '--n_random_starts',
        type=int,
        default=5,
        help='(Bayesian only) Number of random initial evaluations.'
    )

    args = parser.parse_args()

    logger.info("Starting HyperParamTuner CLI with strategy: %s", args.strategy)

    # Initialize tuner
    tuner = HyperParamTuner(
        config_path=args.config,
        evaluation_function=rl_agent_evaluation,
        strategy=args.strategy,
        n_calls=args.n_calls,
        n_random_starts=args.n_random_starts
    )

    # Run tuning pipeline
    best_params = tuner.run_tuning_pipeline()

    logger.info("Hyperparameter tuning complete. Best parameters:")
    logger.info(best_params)

    print("\n Best hyperparameters found:")
    print(best_params)
