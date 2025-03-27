def create_agent(agent_name, config):
    """
    Factory method for creating agents by name.
    """
    agent_name = agent_name.lower()

    if agent_name == "dqn":
        from agents.dqn_agent import DQNAgent
        return DQNAgent(config)

    elif agent_name == "maml":
        from agents.maml_agent import MAMLAgent
        return MAMLAgent(config)

    elif agent_name == "rsi":
        from agents.rsi_agent import RSIAgent
        return RSIAgent(config)

    elif agent_name == "evolution":
        from agents.evolution_agent import EvolutionAgent
        return EvolutionAgent(config=config)

    elif agent_name == "safe_ai":
        from agents.safe_ai_agent import SafeAIAgent
        return SafeAIAgent(config)

    elif agent_name == "autotune":
        from agents.autotune_agent import AutoTuneAgent
        return AutoTuneAgent(config)

    elif agent_name == "multitask":
        from agents.multitask_agent import MultiTaskAgent
        return MultiTaskAgent(config)

    elif agent_name == "cartpole":
        from agents.cartpole_agent import CartpoleAgent
        return CartpoleAgent(config)

    elif agent_name == "collaborative":
        from agents.collaborative_agent import CollaborativeAgent
        return CollaborativeAgent(config)

    else:
        raise ValueError(f"[Agent Factory] Unknown agent name: {agent_name}")
