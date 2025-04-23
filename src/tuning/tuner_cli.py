import argparse
import logging
import json
import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.tuning.tuner import HyperparamTuner
from src.agents.learning.rl_agent import RLAgent

# Setup SLAI logging paths
LOG_FILE = "logs/run.txt"
CHAT_LOG_DIR = "logs/chat_logs"
os.makedirs(CHAT_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

reward_log = []


def rl_evaluation(params):
    """
    Scientific evaluation using SLAI-style RLAgent with grounded reward curves.
    Academic basis:
    - Optimistic Initialization
    - Markov Decision Process
    - TD-learning (Sutton & Barto, 2018)
    """
    env_size = 5
    rewards = []

    # Simplified state-action space
    possible_actions = [0, 1]  # Binary decision space
    agent = RLAgent(possible_actions=possible_actions,
                    learning_rate=params.get("learning_rate", 0.01),
                    epsilon=0.1,
                    discount_factor=0.95,
                    trace_decay=0.8)

    for episode in range(30):
        state = (np.random.randint(env_size),)
        total_reward = 0

        for step in range(15):
            action = agent.step(state)
            reward = np.random.normal(loc=1.0 - abs(state[0] - action), scale=0.5)
            agent.receive_reward(reward)

            next_state = (np.random.randint(env_size),)
            agent.end_episode(next_state, done=(step == 14))
            total_reward += reward
            state = next_state

        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    reward_log.append(avg_reward)
    return avg_reward


def plot_learning_curve():
    plt.figure(figsize=(8, 4))
    plt.plot(reward_log, marker='o')
    plt.title("Average Reward Trend")
    plt.xlabel("Tuning Iteration")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(CHAT_LOG_DIR, "reward_trend.png")
    plt.savefig(path)
    logging.info(f"Plot saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLAI Hyperparameter Tuner via CLI")
    parser.add_argument('--strategy', type=str, choices=['bayesian', 'grid'], required=True,
                        help="Tuning strategy to use")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to existing config file (optional)")
    parser.add_argument('--format', type=str, choices=['json', 'yaml'], default='json',
                        help="Format to use if config is generated dynamically")
    parser.add_argument('--n_calls', type=int, default=20,
                        help="(Bayesian only) Number of optimization calls")
    parser.add_argument('--n_random_starts', type=int, default=5,
                        help="(Bayesian only) Number of random initial evaluations")
    parser.add_argument('--no_generate', action='store_true',
                        help="Prevent auto-generation of config if none provided")

    args = parser.parse_args()
    logging.info("Starting tuner with strategy: %s", args.strategy)

    try:
        tuner = HyperparamTuner(
            config_path=args.config,
            strategy=args.strategy,
            evaluation_function=rl_evaluation,
            n_calls=args.n_calls,
            n_random_starts=args.n_random_starts,
            allow_generate=not args.no_generate,
            config_format=args.format
        )

        best_params = tuner.run_tuning_pipeline()
        print("\n✅ Best hyperparameters found:")
        print(json.dumps(best_params, indent=2))

        plot_learning_curve()

    except Exception as e:
        logging.error("Tuning failed: %s", str(e))
        sys.exit(1)
