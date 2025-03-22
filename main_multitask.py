from tasks.task_sampler import TaskSampler
from agents.multitask_rl import MultiTaskPolicy
import torch
import torch.optim as optim

def main():
    sampler = TaskSampler(['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1'])
    envs = [sampler.sample_task() for _ in range(3)]

    state_size = envs[0][0].observation_space.shape[0]
    action_size = envs[0][0].action_space.n

    multitask_policy = MultiTaskPolicy(state_size, action_size)
    optimizer = optim.Adam(multitask_policy.parameters(), lr=0.001)

    for epoch in range(1000):
        for task_id, (env, name) in enumerate(envs):
            state, _ = env.reset()
            done, truncated = False, False
            total_reward = 0

            while not (done or truncated):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                task_id_tensor = torch.LongTensor([task_id])

                probs = multitask_policy(state_tensor, task_id_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

                next_state, reward, done, truncated, _ = env.step(action)

                # Basic reward signal (expand to REINFORCE / PPO loss)
                total_reward += reward
                state = next_state

            print(f"Task {name} - Reward: {total_reward}")

if __name__ == "__main__":
    main()
