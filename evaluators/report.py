import os
import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(reward_list, title="Reward Over Time", save_path="reports/reward_trend.png"):
    """
    Plot agent reward trend over time.
    
    Args:
        reward_list (list[float]): List of reward values per episode.
        title (str): Plot title.
        save_path (str): File path to save the output image.
    """
    if not reward_list:
        print("[Report] Empty reward list. Skipping plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Report] Reward trend plot saved to: {save_path}")


def plot_success_failure(agent_stats, save_path="reports/success_failure_bar.png"):
    """
    Plot bar chart comparing agent successes vs failures.
    
    Args:
        agent_stats (dict): Format {
            'agent_name': {'successes': int, 'failures': int, ...},
            ...
        }
        save_path (str): Path to save the generated chart.
    """
    if not agent_stats:
        print("[Report] No agent stats available. Skipping success/failure plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    agent_names = list(agent_stats.keys())
    successes = [agent_stats[a].get("successes", 0) for a in agent_names]
    failures = [agent_stats[a].get("failures", 0) for a in agent_names]

    x = np.arange(len(agent_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, successes, width, label="Successes", color="green")
    plt.bar(x + width / 2, failures, width, label="Failures", color="red")

    plt.xlabel("Agents")
    plt.ylabel("Count")
    plt.title("Agent Performance: Successes vs Failures")
    plt.xticks(x, agent_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Report] Success/Failure bar plot saved to: {save_path}")
