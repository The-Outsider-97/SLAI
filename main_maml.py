import os
import sys
import yaml
import time
import torch
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.agent_factory import create_agent
from utils.config_loader import load_config
from utils.logger import setup_logger
from evaluators.report import PerformanceEvaluator
from frontend.main_window import update_visual_output_panel, update_text_output_panel

# Setup logger
logger = setup_logger("MAMLAgent")
evaluator = PerformanceEvaluator(threshold=75.0)


def load_agent_config():
    """Load MAML-specific configuration from YAML."""
    config_path = os.path.join("configs", "agents_config.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        maml_config = config.get("maml", {})
        logger.info("[Config] Loaded MAML config successfully.")
        return maml_config
    except Exception as e:
        logger.error(f"[Config] Failed to load MAML config: {e}")
        sys.exit(1)


def train_and_evaluate(agent):
    """
    Train the MAML agent and evaluate after training.
    Visual output is routed to the frontend panel.
    """
    logger.info("[Training] Starting MAML training...")
    agent.train()

    logger.info("[Evaluation] Running post-training evaluation...")
    results = agent.evaluate()
    score = results.get("average_reward", 0)

    # Evaluate threshold status
    meets = evaluator.meets_threshold(score)
    logger.info(f"[Performance] Average reward: {score} | Meets threshold: {meets}")

    summary = f"Agent: MAML\nAverage Reward: {score}\nMeets Threshold: {meets}"

    # Save visual artifacts
    evaluator.plot_rewards(results.get("reward_trace", []), title="MAML Reward Curve")

    # Frontend updates
    update_text_output_panel(summary)
    update_visual_output_panel(["logs/reward_trend.png", "outputs/reward_trend.png"])

    return results


def main():
    logger.info("=== MAML Agent Execution Start ===")

    maml_config = load_agent_config()
    agent = create_agent("maml", config=maml_config)

    if not torch.cuda.is_available():
        logger.warning("[Warning] CUDA not available. Running on CPU.")

    final_results = train_and_evaluate(agent)

    logger.info("=== MAML Agent Execution Complete ===")
    return final_results


if __name__ == "__main__":
    main()
