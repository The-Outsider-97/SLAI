import os
import sys
import logging
import torch
import time
from collaborative.collaboration_manager import CollaborationManager
from collaborative.shared_memory import SharedMemory

# Import all agents (aligned with your folder and launcher menu)
from agents.evolution_agent import EvolutionaryAgent
from agents.dqn_agent import DQNAgent
from agents.evolutionary_dqn import EvolutionaryDQNAgent
from agents.multitask_rl import MultiTaskRLAgent
from agents.maml_rl import MAMLAgent
from agents.rl_agent import RLAgent

# Assuming you have these elsewhere or placeholder for now
from agents.rsi_agent import RSI_Agent
from agents.safe_ai_agent import SafeAI_Agent

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
    Register all agents to the Collaboration Manager.
    """
    print("\n=== Registering Agents ===")

    # 1. Evolutionary Agent
    collab_mgr.register_agent(
        agent_name="evolutionary",
        agent_class=EvolutionaryAgent(shared_memory),
        capabilities=["optimize", "evolve"]
    )

    # 2. Basic RL Agent (CartPole DQN)
    collab_mgr.register_agent(
        agent_name="dqn",
        agent_class=DQNAgent(shared_memory),
        capabilities=["reinforcement_learning", "decision_making"]
    )

    # 3. Evolutionary DQN Agent
    collab_mgr.register_agent(
        agent_name="evolutionary_dqn",
        agent_class=EvolutionaryDQNAgent(shared_memory),
        capabilities=["evolve", "reinforcement_learning"]
    )

    # 4. Multi-Task RL Agent
    collab_mgr.register_agent(
        agent_name="multitask_rl",
        agent_class=MultiTaskRLAgent(shared_memory),
        capabilities=["multi_task_learning", "adaptation"]
    )

    # 5. Meta-Learning Agent (MAML)
    collab_mgr.register_agent(
        agent_name="maml",
        agent_class=MAMLAgent(shared_memory),
        capabilities=["meta_learning", "fast_adaptation"]
    )

    # 6. Recursive Self-Improvement Agent (RSI)
    collab_mgr.register_agent(
        agent_name="rsi",
        agent_class=RSI_Agent(shared_memory),
        capabilities=["self_improvement", "autotune"]
    )

    # 7. RL Agent (Auto-Tuning)
    collab_mgr.register_agent(
        agent_name="rl_agent",
        agent_class=RLAgent(shared_memory),
        capabilities=["autotune", "reinforcement_learning"]
    )

    # 8. Safe AI Agent
    collab_mgr.register_agent(
        agent_name="safe_ai",
        agent_class=SafeAI_Agent(shared_memory),
        capabilities=["safety", "risk_management"]
    )

    print(f"\nTotal registered agents: {len(collab_mgr.list_agents())}\n")


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
    """
    Main entry point for the collaborative agents system.
    """
    print("\n================ SLAI-v1.5 =================")
    print(" Collaborative Agents & Task Routing System")
    print("===========================================\n")

    # Step 1: Initialize shared memory
    shared_memory = initialize_shared_memory()

    # Step 2: Initialize collaboration manager
    collab_mgr = CollaborationManager()

    # Step 3: Register all agents from SLAI menu
    register_agents(collab_mgr, shared_memory)

    # Step 4: Execute tasks collaboratively
    execute_collaborative_tasks(collab_mgr)

    # Step 5: Inspect shared memory after execution
    display_shared_memory(shared_memory)

    print("\n=============== System Complete ===============\n")


if __name__ == "__main__":
    main()
