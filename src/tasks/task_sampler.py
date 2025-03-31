import gymnasium as gym
import random
import numpy as np
import logging

logger = logging.getLogger('SLAI-TaskSampler')

class TaskSampler:
    """
    TaskSampler for Meta-Learning:
    - Can sample parametric variations of the same environment (gravity, pole length, etc.)
    - Supports standard gym environments and custom ones.
    """

    def __init__(self, base_task='CartPole-v1', num_tasks=10, seed=None):
        """
        Args:
            base_task (str): Gym environment to base tasks on.
            num_tasks (int): Number of different tasks to generate.
            seed (int): Random seed for reproducibility.
        """
        self.base_task = base_task
        self.num_tasks = num_tasks
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        logger.info(f"Initialized TaskSampler with base task: {base_task}")

        # Pre-generate task variations
        self.tasks = self._generate_tasks()

    def _generate_tasks(self):
        """
        Generate a list of tasks with varying parameters.
        E.g., gravity and pole length for CartPole
        """
        tasks = []
        for _ in range(self.num_tasks):
            # Example for CartPole (custom gravity and pole length)
            task_params = {
                'gravity': random.uniform(9.0, 15.0),     # Default 9.8
                'length': random.uniform(0.25, 2.0),      # Default 0.5
                'masscart': random.uniform(0.5, 2.0),     # Default 1.0
                'masspole': random.uniform(0.05, 0.5),    # Default 0.1
                'force_mag': random.uniform(5.0, 15.0)    # Default 10.0
            }
            tasks.append(task_params)

        logger.info(f"Generated {self.num_tasks} task variations.")
        return tasks

    def sample_task(self, return_params=False):
        """
        Sample a single environment with randomized parameters.

        Args:
            return_params (bool): Return the task parameters along with env.

        Returns:
            env (gym.Env): The customized environment instance.
            task_info (dict): Info on environment params if return_params=True.
        """
        task_params = random.choice(self.tasks)

        # Create a CartPole environment (or extend this for other envs)
        env = gym.make(self.base_task)

        # Modify environment parameters directly
        if hasattr(env, 'env'):
            env_obj = env.env  # Classic gym
        else:
            env_obj = env.unwrapped  # Newer gymnasium

        # Apply parameters (this depends on environment's internal attributes)
        try:
            env_obj.gravity = task_params['gravity']
            env_obj.length = task_params['length']
            env_obj.masscart = task_params['masscart']
            env_obj.masspole = task_params['masspole']
            env_obj.force_mag = task_params['force_mag']
        except AttributeError as e:
            logger.error(f"Failed to apply task params: {task_params}. Error: {e}")
            raise

        logger.debug(f"Sampled task with params: {task_params}")

        if return_params:
            return env, task_params
        else:
            return env, None
