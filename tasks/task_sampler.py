import gymnasium as gym
import random

class TaskSampler:
    def __init__(self, tasks=None):
        # List of environment names
        self.tasks = tasks or ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']

    def sample_task(self):
        env_name = random.choice(self.tasks)
        env = gym.make(env_name)
        return env, env_name
