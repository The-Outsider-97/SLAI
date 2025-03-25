import random
import copy
import numpy as np
from agents.dqn_agent import DQNAgent
import torch

class EvolutionaryTrainer:
    def __init__(self, env, population_size=10, generations=10, mutation_rate=0.2):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []

    def random_config(self):
        """Generate a random hyperparameter configuration"""
        return {
            'gamma': np.random.uniform(0.90, 0.999),  
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': np.random.uniform(0.990, 0.999),
            'learning_rate': np.random.uniform(0.0001, 0.01),
            'hidden_size': random.choice([32, 64, 128])
        }

    def initialize_population(self, state_size, action_size):
        """Initialize a population of agents with random configurations"""
        self.population = [DQNAgent(state_size, action_size, self.random_config()) for _ in range(self.population_size)]

    def evaluate_agent(self, agent, episodes=5):
        """Evaluate agent by running episodes and averaging rewards"""
        total_rewards = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            for _ in range(500): 
                action = agent.select_action(state, explore=False)
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                if done or truncated:
                    break
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

    def mutate(self, config):
        """Mutate a hyperparameter configuration"""
        mutated_config = copy.deepcopy(config)
        if np.random.rand() < self.mutation_rate:
            mutated_config['gamma'] = np.clip(mutated_config['gamma'] + np.random.uniform(-0.02, 0.02), 0.90, 0.999)
        if np.random.rand() < self.mutation_rate:
            mutated_config['learning_rate'] = np.clip(mutated_config['learning_rate'] * np.random.uniform(0.5, 1.5), 0.0001, 0.01)
        if np.random.rand() < self.mutation_rate:
            mutated_config['epsilon_decay'] = np.clip(mutated_config['epsilon_decay'] + np.random.uniform(-0.005, 0.005), 0.990, 0.999)
        return mutated_config

    def evolve(self, state_size, action_size):
        """Run evolution for multiple generations"""
        self.initialize_population(state_size, action_size)

        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")

            # Evaluate all agents
            performances = [(agent, self.evaluate_agent(agent)) for agent in self.population]
            performances.sort(key=lambda x: x[1], reverse=True)

            # Select top 50% as parents
            num_elites = self.population_size // 2
            parents = [p[0] for p in performances[:num_elites]]

            # Mutate and create next generation
            new_population = parents.copy()
            while len(new_population) < self.population_size:
                parent = random.choice(parents)
                mutated_config = self.mutate(parent.config)
                new_population.append(DQNAgent(state_size, action_size, mutated_config))

            self.population = new_population

            # Best agent from this generation
            best_agent, best_score = performances[0]
            print(f"Best agent score: {best_score}")

        return best_agent

class EvolutionaryDQNAgent:
    def __init__(self, env, state_size, action_size):
        self.trainer = EvolutionaryTrainer(env)
        self.state_size = state_size
        self.action_size = action_size

    def execute(self, task_data):
        agent = self.trainer.evolve(state_size=self.state_size, action_size=self.action_size)
        return {
            "status": "success",
            "agent": "EvolutionaryDQNAgent",
            "best_agent_config": agent.config
        }
