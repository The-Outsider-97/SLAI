import argparse
import logging
from hyperparam_tuning.tuner import HyperParamTuner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Real evaluation function example (plug in your own model logic here)
def rl_agent_evaluation(params):
    """
    Realistic evaluation function for an RL agent.
    Adjust this function to fit your actual model and data pipeline.

    Args:
        params (dict): Hyperparameters to evaluate.

    Returns:
        float: Performance score (higher is better).
    """
    learning_rate = params['learning_rate']
    num_layers = params['num_layers']
    activation_function = params['activation']

    logger.info(f"Evaluating with learning_rate={learning_rate}, num_layers={num_layers}, activation={activation_function}")

    # === Replace this with your model training + evaluation pipeline ===
    # Example pseudo-logic:
    # agent = RLAgent(learning_rate, num_layers, activation_function)
    # agent.train(episodes=100)
    # reward = agent.evaluate()
    # ================================================================

    # Dummy reward simulation for demo purposes
    reward = -((learning_rate - 0.01) ** 2 + (num_layers - 3) ** 2)
    logger.info(f"Achieved reward: {reward}")

    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hyperparameter Tuning")

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
        help='Tuning strategy to use: bayesian or grid.'
    )

    parser.add_argument(
        '--n_calls',
        type=int,
        default=20,
        help='Number of calls for Bayesian optimization (ignored in grid).'  
    )

    parser.add_argument(
        '--n_random_starts',
        type=int,
        default=5,
        help='Number of random starts for Bayesian optimization (ignored in grid).'
    )

    args = parser.parse_args()

    logger.info("Starting HyperParamTuner CLI...")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Strategy: {args.strategy}")

    tuner = HyperParamTuner(
        config_path=args.config,
        evaluation_function=rl_agent_evaluation,  # Replace with your own model eval function!
        strategy=args.strategy,
        n_calls=args.n_calls,
        n_random_starts=args.n_random_starts
    )

    best_params = tuner.run_tuning_pipeline()
    print("\nBest hyperparameters found:")
    print(best_params)
