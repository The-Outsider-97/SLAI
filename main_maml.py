import os
import sys
import yaml
import torch
import logging
import time
import tempfile
from agents.maml_rl import MAMLAgent
from utils.logger import setup_logger
from logs.logger import get_logger
from tasks.task_sampler import TaskSampler

# Logger setup
logger = setup_logger("MAMLAgent", level=logging.INFO)

# === Load configuration ===
config_path = "config.yaml"
if not os.path.exists(config_path):
    logger.error(f"Configuration file not found: {config_path}")
    sys.exit(1)

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Error reading config.yaml: {e}")
    sys.exit(1)

logger.info("Configuration loaded successfully.")

# === Extract and validate required config values ===
try:
    agent_config = config.get("agent", {})
    training_config = config.get("training", {})

    state_size = agent_config["state_size"]
    action_size = agent_config["action_size"]
except KeyError as e:
    logger.error(f"Missing key in config.yaml under 'agent': {e}")
    sys.exit(1)

# Training parameters with defaults if not present
iterations = training_config.get("iterations", 100)
tasks_per_iteration = training_config.get("tasks_per_iteration", 5)
log_interval = training_config.get("log_interval", 10)

logger.info(f"Agent params — state_size: {state_size}, action_size: {action_size}")
logger.info(f"Training setup — iterations: {iterations}, tasks_per_iteration: {tasks_per_iteration}, log_interval: {log_interval}")

# === Define MAMLAgent with train() method if not already implemented ===
class MAMLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Initialize model, optimizer, etc. here

    def train(self, tasks):
        # Placeholder logic
        for task in tasks:
            print(f"Training on task: {task}")

# === Initialize MAML agent ===
try:
    agent = MAMLAgent(state_size=state_size, action_size=action_size)
    logger.info("MAMLAgent initialized.")
except Exception as e:
    logger.error(f"Failed to initialize MAMLAgent: {e}", exc_info=True)
    sys.exit(1)

# === Initialize task sampler ===
try:
    task_sampler = TaskSampler()
    logger.info("TaskSampler initialized.")
except Exception as e:
    logger.error(f"Failed to initialize TaskSampler: {e}", exc_info=True)
    sys.exit(1)

# === Training Loop ===
try:
    for iteration in range(1, iterations + 1):
        tasks = task_sampler.sample_task(tasks_per_iteration)
        agent.train(tasks)

        if iteration % log_interval == 0 or iteration == 1:
            logger.info(f"[Iteration {iteration}] Training in progress...")

    logger.info("MAML training completed successfully.")

except Exception as e:
    logger.error(f"Error during MAML training loop: {e}", exc_info=True)
    sys.exit(1)
