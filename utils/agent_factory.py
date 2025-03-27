def create_agent(agent_name, config):
    """
    Factory method for creating agents by name.
    """
    agent_name = agent_name.lower()

    if agent_name == "dqn":
        from agents.dqn_agent import DQNAgent
        return DQNAgent(
            state_size=config["state_size"],
            action_size=config["action_size"],
            config=config
        )

    elif agent_name == "evolution":
        from agents.evolution_agent import EvolutionAgent
        return EvolutionAgent(
            input_size=config.get("input_size", 10),
            output_size=config.get("output_size", 2),
            config=config
        )

    elif agent_name == "evolutionary_dqn":
        from agents.evolutionary_dqn import EvolutionaryDQNAgent
        import gym
        env = gym.make(config.get("env", "CartPole-v1"))
        return EvolutionaryDQNAgent(
            env=env,
            state_size=config["state_size"],
            action_size=config["action_size"]
        )

    elif agent_name == "multitask":
        from agents.multitask_rl import MultiTaskRLAgent
        return MultiTaskRLAgent(
            state_size=config["state_size"],
            action_size=config["action_size"],
            num_tasks=config["num_tasks"]
        )

    elif agent_name == "maml":
        from agents.maml_rl import MAMLAgent
        return MAMLAgent(
            state_size=config["state_size"],
            action_size=config["action_size"],
            hidden_size=config["hidden_size"],
            meta_lr=config["meta_lr"],
            inner_lr=config["inner_lr"],
            gamma=config["gamma"]
        )

    elif agent_name == "rsi":
        from agents.rsi_agent import RSI_Agent
        return RSI_Agent(
            state_size=config["state_size"],
            action_size=config["action_size"],
            shared_memory=config["shared_memory"]
        )

    elif agent_name == "rl_agent":
        from agents.rl_agent import RLAgent
        return RLAgent(
            learning_rate=config["learning_rate"],
            num_layers=config["num_layers"],
            activation_function=config["activation_function"]
        )

    elif agent_name == "safe_ai":
        from agents.safe_ai_agent import SafeAI_Agent
        return SafeAI_Agent(shared_memory=config["shared_memory"])

    elif agent_name == "collaborative":
        from agents.collaborative_agent import CollaborativeAgent
        return CollaborativeAgent(
            shared_memory=config["shared_memory"],
            task_router=config["task_router"]
        )

    else:
        raise ValueError(f"[Agent Factory] Unknown agent name: {agent_name}")
