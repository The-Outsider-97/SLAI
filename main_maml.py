import torch
import logging
import time
import tempfile
import sys
import os
from logger import get_logger
from tasks.task_sampler import TaskSampler
from agents.maml_rl import MAMLAgent
from utils.logger import setup_logger

def self_modify_and_restart(new_code):
    # Save a backup
    backup_path = __file__ + '.bak'
    with open(backup_path, 'w') as backup_file:
        with open(__file__, 'r') as original_file:
            backup_file.write(original_file.read())

    # Validate syntax before replacing
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        temp_file.write(new_code)
        temp_path = temp_file.name

    try:
        compile(open(temp_path).read(), __file__, 'exec')
    except SyntaxError as e:
        print(f"Syntax error in new code! Aborting rewrite.\n{e}")
        return

    # If validation passes, overwrite the file
    with open(__file__, 'w') as f:
        f.write(new_code)

    # Restart
    os.execv(sys.executable, ['python'] + sys.argv)

# ===============================
# Initialize Logger
# ===============================
logger = setup_logger('SLAI-MAML', level=logging.DEBUG)

# ===============================
# Meta-Learning Training Function
# ===============================
def meta_train_maml(config=None):
    # ===============================
    # Default Configuration
    # ===============================
    if config is None:
        config = {
            'base_task': 'CartPole-v1',
            'num_tasks': 10,
            'meta_iterations': 500,
            'tasks_per_meta_update': 4,
            'eval_interval': 50,
            'max_steps_per_task': 200,
            'hidden_size': 64,
            'meta_lr': 0.001,
            'inner_lr': 0.01,
            'gamma': 0.99,
            'seed': 42
        }

    logger.info(f"Starting MAML Meta-Learning with config: {config}")

    # ===============================
    # Task Sampler Initialization
    # ===============================
    sampler = TaskSampler(
        base_task=config['base_task'],
        num_tasks=config['num_tasks'],
        seed=config['seed']
    )

    logger.info(f"Initialized Task Sampler for {config['num_tasks']} tasks on {config['base_task']}")

    # Sample one task to get state/action sizes
    env, task_params = sampler.sample_task(return_params=True)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    logger.info(f"Sampled task for initialization with params: {task_params}")
    logger.info(f"Sampled Task: {config['base_task']}, State size: {state_size}, Action size: {action_size}")

    env.close()

    # ===============================
    # Initialize MAML Agent
    # ===============================
    maml_agent = MAMLAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=config['hidden_size'],
        meta_lr=config['meta_lr'],
        inner_lr=config['inner_lr'],
        gamma=config['gamma']
    )

    logger.info("Initialized MAML Agent.")

    # ===============================
    # Meta-Training Loop
    # ===============================
    for iteration in range(1, config['meta_iterations'] + 1):
        start_time = time.time()

        # Sample tasks for this meta-update
        tasks = [sampler.sample_task(return_params=True) for _ in range(config['tasks_per_meta_update'])]

        envs = [(env, params) for env, params in tasks]

        # Extract environments for training
        env_list = [env for env, _ in envs]

        # Perform a meta-update
        meta_loss = maml_agent.meta_update(envs)  # pass the list of tuples

        elapsed_time = time.time() - start_time

        logger.info(
            f"[Meta Iter {iteration}/{config['meta_iterations']}] Meta-Loss: {meta_loss:.6f} | Time: {elapsed_time:.2f}s"
        )

        # Evaluate periodically
        if iteration % config['eval_interval'] == 0:
            eval_meta_policy(maml_agent, sampler)

    logger.info("Meta-Training Completed.")


# ===============================
# Optional Evaluation Function
# ===============================
def eval_meta_policy(maml_agent, sampler, episodes=5):
    logger.info("Evaluating Meta-Learned Policy on Random Tasks...")

    env, task_params = sampler.sample_task(return_params=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    logger.info(f"Evaluation Task: {sampler.base_task} with params: {task_params}")

    policy = maml_agent.policy
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action, _ = maml_agent.get_action(state, policy=policy)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done or truncated:
                break

        rewards.append(total_reward)
        logger.info(f"Eval Episode {episode+1}/{episodes} | Reward: {total_reward}")

    avg_reward = sum(rewards) / episodes
    logger.info(f"Average Reward on {sampler.base_task} with params {task_params}: {avg_reward:.2f}")

    env.close()


# ===============================
# Main Function Entry Point
# ===============================
if __name__ == "__main__":
    # Example configuration (customizable or from config.yaml)
    maml_config = {
        'base_task': 'CartPole-v1',
        'num_tasks': 10,
        'meta_iterations': 500,
        'tasks_per_meta_update': 4,
        'eval_interval': 50,
        'max_steps_per_task': 200,
        'hidden_size': 64,
        'meta_lr': 0.001,
        'inner_lr': 0.01,
        'gamma': 0.99,
        'seed': 42
    }

    meta_train_maml(config=maml_config)
