import sys
import logging
import torch
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collaboration_manager import CollaborationManager
from shared_memory import SharedMemory

# Import all agents (aligned with your folder and launcher menu)
from agents.evolution_agent import EvolutionAgent
from agents.dqn_agent import DQNAgent
from agents.evolutionary_dqn import EvolutionaryDQNAgent
from agents.multitask_rl import MultiTaskRLAgent
from agents.maml_rl import MAMLAgent
from agents.rl_agent import RLAgent

# Assuming you have these elsewhere or placeholder for now
from agents.rsi_agent import RSI_Agent
from agents.safe_ai_agent import SafeAI_Agent

# 1. Run safety agent
def run_safety_protocol(collab_mgr):
    print("\n=== Safety Protocol Check ===")
    task_data = {
        "policy_risk_score": 0.27,
        "task_type": "reinforcement_learning"
    }
    safe_result = collab_mgr.run_task("safety", task_data)
    print("SafeAI Task Output:", safe_result)

# 2. Train SafeAI Agent from historical data
safe_ai_agent.train()

# 3. Evaluate performance after training
summary = safe_ai_agent.evaluate()
print(summary)

def initialize_shared_memory():
    """
    Initialize shared memory with global variables that can be accessed by all agents.
    """
    print("\n=== Initializing Shared Memory ===")
    shared_memory = SharedMemory()

    # Pre-populate shared knowledge or context
    shared_memory.set("global_best_score", 0.50)
    shared_memory.set("current_task_id", 0)
    shared_memory.set("knowledge_base", {
        "cartpole": {"baseline_reward": 200},
        "multitask": {"domains": ["CartPole", "MountainCar"]},
        "safety_guidelines": {"max_risk": 0.2}
    })

    return shared_memory


def register_agents(collab_mgr, shared_memory):
    """
    Register all SLAI agents into the collaborative system.
    Includes real parameters and placeholder configs.
    """
    print("\n=== Registering Agents ===")

    # === 1. Evolutionary Agent ===
    evolutionary_agent = EvolutionAgent(
        input_size=10,
        output_size=2,
        config={
            "hidden_sizes": [32, 64, 128],
            "learning_rate": 0.001,
            "population_size": 5,
            "elite_fraction": 0.4
        }
    )
    collab_mgr.register_agent(
        agent_name="evolutionary",
        agent_class=evolutionary_agent,
        capabilities=["optimize", "evolve"]
    )

    # === 2. DQN Agent ===
    dqn_agent = DQNAgent(
        state_size=4,
        action_size=2,
        config={
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 1e-3,
            "hidden_size": 128,
            "batch_size": 64,
            "memory_capacity": 10000
        }
    )
    collab_mgr.register_agent(
        agent_name="dqn",
        agent_class=dqn_agent,
        capabilities=["reinforcement_learning", "decision_making"]
    )

    # === 3. Evolutionary DQN Agent ===
    from gym.envs.classic_control import CartPoleEnv
    env = CartPoleEnv()

    evolutionary_dqn_agent = EvolutionaryDQNAgent(
        env=env,
        state_size=4,
        action_size=2
    )
    collab_mgr.register_agent(
        agent_name="evolutionary_dqn",
        agent_class=evolutionary_dqn_agent,
        capabilities=["evolve", "reinforcement_learning"]
    )

    # === 4. Multi-Task RL Agent ===
    multitask_rl_agent = MultiTaskRLAgent(
        state_size=4,
        action_size=2,
        num_tasks=10
    )
    collab_mgr.register_agent(
        agent_name="multitask_rl",
        agent_class=multitask_rl_agent,
        capabilities=["multi_task_learning", "adaptation"]
    )

    # === 5. MAML Agent ===
    maml_agent = MAMLAgent(
        state_size=4,
        action_size=2,
        hidden_size=64,
        meta_lr=0.001,
        inner_lr=0.01,
        gamma=0.99
    )
    collab_mgr.register_agent(
        agent_name="maml",
        agent_class=maml_agent,
        capabilities=["meta_learning", "fast_adaptation"]
    )

    # === 6. Recursive Self-Improvement (RSI) Agent ===
    from agents.rsi_agent import RSI_Agent  # You’ll need to implement or wrap this
    rsi_agent = RSI_Agent(shared_memory)
    collab_mgr.register_agent(
        agent_name="rsi",
        agent_class=rsi_agent,
        capabilities=["self_improvement", "autotune"]
    )

    # === 7. RL Agent ===
    rl_agent = RLAgent(
        learning_rate=0.01,
        num_layers=2,
        activation_function="relu"
    )
    collab_mgr.register_agent(
        agent_name="rl_agent",
        agent_class=rl_agent,
        capabilities=["autotune", "reinforcement_learning"]
    )

    # === 8. Safe AI Agent ===
    from agents.safe_ai_agent import SafeAI_Agent  # Also needs implementation
    safe_ai_agent = SafeAI_Agent(shared_memory)
    collab_mgr.register_agent(
        agent_name="safe_ai",
        agent_class=safe_ai_agent,
        capabilities=["safety", "risk_management"]
    )

    print(f"\nTotal registered agents: {len(collab_mgr.list_agents())}")
    for name in collab_mgr.list_agents():
        print(f" - {name}")


def execute_collaborative_tasks(collab_mgr):
    """
    Demonstrate collaborative task execution.
    """
    print("\n=== Executing Collaborative Tasks ===")

    # Define a task queue covering all major capabilities
    task_queue = [
        {"type": "optimize", "data": {"dataset": "CartPole-v1", "current_score": 0.50}},
        {"type": "reinforcement_learning", "data": {"state": "start", "episode": 1}},
        {"type": "evolve", "data": {"population_size": 50, "generations": 10}},
        {"type": "multi_task_learning", "data": {"domains": ["CartPole", "MountainCar"]}},
        {"type": "meta_learning", "data": {"tasks": ["few_shot_learning"]}},
        {"type": "self_improvement", "data": {"system_health": 90}},
        {"type": "autotune", "data": {"hyperparameters": {"lr": 0.01, "batch_size": 32}}},
        {"type": "safety", "data": {"policy_risk_score": 0.15}}
    ]

    for idx, task in enumerate(task_queue, 1):
        task_type = task["type"]
        task_data = task["data"]

        print(f"\n--- Task {idx}: {task_type} ---")

        try:
            result = collab_mgr.run_task(task_type, task_data)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Task {task_type} failed: {str(e)}")

        time.sleep(1)  # Simulate processing delay


def display_shared_memory(shared_memory):
    """
    After task execution, display the final state of shared memory.
    """
    print("\n=== Final Shared Memory ===")

    keys = shared_memory.keys()

    for key in keys:
        value = shared_memory.get(key)
        print(f"{key}: {value}")


def main():
    print("\n================ SLAI-v1.5 =================")
    print(" Collaborative Agents & Task Routing System")
    print("===========================================\n")

    # Step 1: Initialize shared memory
    shared_memory = initialize_shared_memory()

    #Safe AI pre-population
    shared_memory.set("safe_ai_recommendation", {
        "risk_threshold": 0.2,  # max acceptable risk level
        "parity_diff_threshold": 0.1,  # fairness threshold
        "tpr_diff_threshold": 0.1,  # true positive rate threshold
        "last_violation": {
            "metric": None,
            "value": None,
            "timestamp": None
        },
        "violation_history": [],  # track all violations
        "actions_taken": [],  # ['rollback', 'hyperparam_tune', 'agent_switch']
        "recommended_action": None,  # current system suggestion
        "evaluated_agents": {},  # agent_id: {"score": float, "bias": float}
        "performance_metrics": {
            "last_reward": None,
            "fairness_score": None
        },
        "safe_config_backup": None  # pointer to last known safe config
    })

    # Step 2: Initialize collaboration manager
    collab_mgr = CollaborationManager()

    # Step 3: Register all agents
    register_agents(collab_mgr, shared_memory)

    # Step 3.5: Run safety protocol task explicitly
    run_safety_protocol(collab_mgr)

    # Step 4: Demonstrate shared memory agent status tracking
    from registry import AgentRegistry
    class DemoAgent:
        def execute(self, data): return f"DemoAgent executed {data}"
    registry = AgentRegistry(shared_memory=shared_memory)
    registry.register("demo_agent", DemoAgent(), ["routing", "diagnostics"])
    registry.update_status("demo_agent", "busy")
    print(f"[SharedMemory] demo_agent status: {registry.get_status('demo_agent')}")

    # Step 5: Execute collaborative tasks
    execute_collaborative_tasks(collab_mgr)

    # Step 6: Inspect shared memory
    display_shared_memory(shared_memory)

    print("\n=============== System Complete ===============\n")

if __name__ == "__main__":
    main()
