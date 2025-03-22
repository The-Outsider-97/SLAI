from tasks.task_sampler.py import TaskSampler
from agents.maml_rl import MAMLAgent
import torch

def main():
    sampler = TaskSampler(['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1'])

    # Assume all tasks have the same state/action space size for now
    env, _ = sampler.sample_task()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    maml_agent = MAMLAgent(state_size, action_size)

    meta_iterations = 1000
    tasks_per_meta_update = 4

    for iteration in range(meta_iterations):
        tasks = [sampler.sample_task() for _ in range(tasks_per_meta_update)]
        meta_loss = maml_agent.meta_update(tasks)

        if iteration % 10 == 0:
            print(f"Meta-Iteration {iteration}, Meta-Loss: {meta_loss:.4f}")

if __name__ == "__main__":
    main()
