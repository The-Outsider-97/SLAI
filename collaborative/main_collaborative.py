import sys
import logging
import torch
import time
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collaboration_manager import CollaborationManager
from shared_memory import SharedMemory
from registry import AgentRegistry
from utils.logger import setup_logger

# Import all agents
from collaborative_agent import CollaborativeAgent
from agents.evolution_agent import EvolutionAgent
from agents.dqn_agent import DQNAgent
from agents.evolutionary_dqn import EvolutionaryDQNAgent
from agents.multitask_rl import MultiTaskRLAgent
from agents.maml_rl import MAMLAgent
from agents.rl_agent import RLAgent
from agents.rsi_agent import RSI_Agent
from agents.safe_ai_agent import SafeAI_Agent

logger = setup_logger("CollaborativeAgent", level=logging.INFO)

try:
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load config.yaml: {e}")
    sys.exit(1)

def run_safety_protocol(agent):
    print("\n=== Safety Protocol Check ===")
    task_type = "safety"
    task_data = {
        "policy_risk_score": 0.27,
        "task_type": "reinforcement_learning"
    }
    result = agent.execute(task_type, task_data)
    print("SafeAI Task Output:", result)

def initialize_shared_memory():
    print("\n=== Initializing Shared Memory ===")
    shared_memory = SharedMemory()
    shared_memory.set("global_best_score", 0.50)
    shared_memory.set("current_task_id", 0)
    shared_memory.set("knowledge_base", {
        "cartpole": {"baseline_reward": 200},
        "multitask": {"domains": ["CartPole", "MountainCar"]},
        "safety_guidelines": {"max_risk": 0.2}
    })
    return shared_memory

def register_agents(collab_mgr, shared_memory):
    print("\n=== Registering Agents ===")

    evolutionary_agent = EvolutionAgent(input_size=10, output_size=2, config={
        "hidden_sizes": [32, 64, 128],
        "learning_rate": 0.001,
        "population_size": 5,
        "elite_fraction": 0.4
    })
    collab_mgr.register_agent("evolutionary", evolutionary_agent, ["optimize", "evolve"])

    dqn_agent = DQNAgent(state_size=4, action_size=2, config={
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "learning_rate": 1e-3,
        "hidden_size": 128,
        "batch_size": 64,
        "memory_capacity": 10000
    })
    collab_mgr.register_agent("dqn", dqn_agent, ["reinforcement_learning", "decision_making"])

    from gym.envs.classic_control import CartPoleEnv
    env = CartPoleEnv()
    evolutionary_dqn_agent = EvolutionaryDQNAgent(env=env, state_size=4, action_size=2)
    collab_mgr.register_agent("evolutionary_dqn", evolutionary_dqn_agent, ["evolve", "reinforcement_learning"])

    multitask_rl_agent = MultiTaskRLAgent(state_size=4, action_size=2, num_tasks=10)
    collab_mgr.register_agent("multitask_rl", multitask_rl_agent, ["multi_task_learning", "adaptation"])

    maml_agent = MAMLAgent(state_size=4, action_size=2, hidden_size=64, meta_lr=0.001, inner_lr=0.01, gamma=0.99)
    collab_mgr.register_agent("maml", maml_agent, ["meta_learning", "fast_adaptation"])

    rsi_agent = RSI_Agent(state_size=4, action_size=2, shared_memory=shared_memory)
    collab_mgr.register_agent("rsi", rsi_agent, ["self_improvement", "autotune"])

    rl_agent = RLAgent(learning_rate=0.01, num_layers=2, activation_function="relu")
    collab_mgr.register_agent("rl_agent", rl_agent, ["autotune", "reinforcement_learning"])

    safe_ai_agent = SafeAI_Agent(shared_memory=shared_memory)
    collab_mgr.register_agent("safe_ai", safe_ai_agent, ["safety", "risk_management"])

    print(f"\nTotal registered agents: {len(collab_mgr.list_agents())}")
    for name in collab_mgr.list_agents():
        print(f" - {name}")

    return safe_ai_agent

def execute_collaborative_tasks(agent):
    print("\n=== Executing Collaborative Tasks ===")
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
        print(f"\n--- Task {idx}: {task['type']} ---")
        try:
            result = agent.execute(task['type'], task['data'])
            print(f"Result: {result}")
        except Exception as e:
            print(f"Task {task['type']} failed: {str(e)}")
        time.sleep(1)

def display_shared_memory(shared_memory):
    print("\n=== Final Shared Memory ===")
    for key in shared_memory.keys():
        print(f"{key}: {shared_memory.get(key)}")

def main():
    print("\n================ SLAI-v1.5 =================")
    print(" Collaborative Agents & Task Routing System")
    print("===========================================\n")

    shared_memory = initialize_shared_memory()

    from task_router import TaskRouter
    from collaboration_manager import CollaborationManager

    collab_mgr = CollaborationManager(shared_memory=shared_memory)
    task_router = collab_mgr.router

    register_agents(collab_mgr, shared_memory)

    collab_agent = CollaborativeAgent(shared_memory, task_router)
    logger.info("CollaborativeAgent initialized.")

    run_safety_protocol(collab_agent)

    print("\n=== Direct SafeAI Agent Training ===")
    safe_ai_agent = collab_mgr.registry.get_agent_class("safe_ai")
    safe_ai_agent.train()
    summary = safe_ai_agent.evaluate()
    print("Evaluation Summary:")
    print(summary)

    class DemoAgent:
        def execute(self, data): return f"DemoAgent executed {data}"

    registry = AgentRegistry(shared_memory=shared_memory)
    registry.register("demo_agent", DemoAgent(), ["routing", "diagnostics"])
    registry.update_status("demo_agent", "busy")
    print(f"[SharedMemory] demo_agent status: {registry.get_status('demo_agent')}")

    execute_collaborative_tasks(collab_agent)
    display_shared_memory(shared_memory)

    print("\n=============== System Complete ===============\n")

if __name__ == "__main__":
    main()
