import time
import os, sys
import json
import logging
import argparse
from utils.agent_factory import create_agent
from utils.logger import setup_logger
from tasks.task_sampler import TaskSampler
from evaluators.report import PerformanceEvaluator
from frontend.main_window import update_visual_output_panel

# ===============================
# Initialize Logger
# ===============================
logger = setup_logger('SLAI-MultiTask', level='INFO')

# ===============================
# Main Function Entry Point
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    args = parser.parse_args()

    # Configuration for multitask agent
    multitask_config = {
        'tasks': ['CartPole-v1'],
        'num_tasks': 3,
        'task_embedding_size': 16,
        'hidden_size': 128,
        'lr': 0.001,
        'epochs': args.episodes,
        'max_timesteps': 500,
        'gamma': 0.99
    }

    # Use TaskSampler to dynamically extract state/action dimensions
    sampler = TaskSampler(
        base_task=multitask_config['tasks'][0],
        num_tasks=multitask_config['num_tasks']
    )
    env, _ = sampler.sample_task(return_params=True)
    multitask_config['state_size'] = env.observation_space.shape[0]
    multitask_config['action_size'] = env.action_space.n

    agent = create_agent("multitask", config=multitask_config)
    evaluator = PerformanceEvaluator()
    reward_trace = []
    stats = {"successes": 0, "failures": 0}

    print("Training MultiTask RL Agent...")
    for episode in range(args.episodes):
        result = agent.execute({"episode": episode})
        print(f"Episode {episode}: {result}")
        if isinstance(result, dict):
            if "reward" in result:
                reward_trace.append(result["reward"])
            if result.get("status") == "success":
                stats["successes"] += 1
            else:
                stats["failures"] += 1

    reward_plot_path = "logs/reward_trend.png"
    bar_plot_path = "logs/success_failure_bar.png"

    evaluator.plot_rewards(reward_trace, title="MultiTask Agent Reward Over Time", save_path=reward_plot_path)
    evaluator.plot_success_failure({"MultiTaskAgent": stats}, save_path=bar_plot_path)

    summary_data = {
        "reward_trace": reward_trace,
        "agent_stats": stats
    }

    with open("logs/eval_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Send result to SLAI frontend panels
    update_visual_output_panel(image_paths=[reward_plot_path, bar_plot_path])

    text_summary = f"Total Episodes: {args.episodes}\nSuccesses: {stats['successes']}\nFailures: {stats['failures']}\n"
    if reward_trace:
        text_summary += f"Average Reward: {sum(reward_trace) / len(reward_trace):.2f}\n"
        text_summary += f"Max Reward: {max(reward_trace):.2f}\nMin Reward: {min(reward_trace):.2f}"
    update_visual_output_panel(text_summary)

if __name__ == "__main__":
    main()
