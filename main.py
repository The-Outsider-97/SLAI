import os
import sys
import yaml
import torch
import logging
import threading
import subprocess

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logs.logger import get_logger
from utils.logger import setup_logger
from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
from torch.utils.data import DataLoader, TensorDataset
from frontend.main import launch_ui

launch_ui()

# ============================
# SETUP LOGGER FIRST
# ============================
logger = setup_logger('SLAI', level=logging.DEBUG)

# ============================
# LOAD CONFIGURATION
# ============================
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    logger.error("config.yaml not found. Make sure the file exists in the working directory.")
    sys.exit(1)
except yaml.YAMLError as e:
    logger.error(f"Error parsing config.yaml: {e}")
    sys.exit(1)

# ============================
# VALIDATE CONFIG STRUCTURE
# ============================
required_keys = [
    ('agent', ['input_size', 'output_size']),
    ('training', ['num_samples']),
    ('evaluator', ['threshold'])
]

for section, keys in required_keys:
    if section not in config:
        logger.error(f"Missing section '{section}' in config.yaml.")
        sys.exit(1)
    for key in keys:
        if key not in config[section]:
            logger.error(f"Missing key '{key}' in section '{section}' of config.yaml.")
            sys.exit(1)

# ============================
# DEVICE CONFIGURATION
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device in use: {device}")

# ============================
# DATA GENERATION FUNCTION
# ============================
def generate_dummy_data(num_samples=500, input_size=10, output_size=2):
    X = torch.randn(num_samples, input_size, device=device)
    y = torch.randint(0, output_size, (num_samples,), device=device)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)

# ============================
# EVOLUTIONARY AGENT RUNNER
# ============================
def evolutionary_agent_run():
    os.makedirs('logs', exist_ok=True)
    logger.info("Starting SLAI Evolutionary Agent...")

    input_size = config['agent']['input_size']
    output_size = config['agent']['output_size']

    agent = EvolutionAgent(input_size=input_size, output_size=output_size, config=config['agent'])
    evaluator = PerformanceEvaluator(threshold=config['evaluator']['threshold'])

    best_performance = 0
    best_model = None

    logger.info("Generating training and validation data.")
    train_loader = generate_dummy_data(
        num_samples=config['training']['num_samples'],
        input_size=input_size,
        output_size=output_size
    )
    val_loader = generate_dummy_data(
        num_samples=config['training']['num_samples'],
        input_size=input_size,
        output_size=output_size
    )

    agent.initialize_population()

    for generation in range(10):
        logger.info(f"Starting Evolutionary Generation {generation + 1}")

        try:
            agent.evolve_population(evaluator, train_loader, val_loader)

            best_model = agent.population[0]['model']
            best_performance = agent.population[0]['performance']

            logger.info(f"Evolved Best Model - Gen {generation + 1}: {best_performance:.2f}% accuracy")
            torch.save(best_model.state_dict(), f'logs/best_model_evolved_gen_{generation + 1}.pth')

        except Exception as e:
            logger.error(f"Evolution failed in Generation {generation + 1}: {e}", exc_info=True)

    logger.info("Running manual experiment on custom model...")

    try:
        manual_model = agent.build_model(hidden_size=64).to(device)
        logger.debug(f"Manual model architecture: {manual_model}")

        agent.train_model({'model': manual_model, 'learning_rate': 0.001}, train_loader, val_loader)
        manual_performance = agent.evaluate_model({'model': manual_model}, val_loader)

        logger.info(f"Manual model accuracy: {manual_performance:.2f}%")

        if evaluator.is_better(manual_performance, best_performance):
            logger.info(f"Manual model outperformed evolved models! Accuracy: {manual_performance:.2f}%")
            best_model = manual_model
            best_performance = manual_performance
            torch.save(manual_model.state_dict(), 'logs/best_manual_model.pth')

    except Exception as e:
        logger.error(f"Manual experiment failed: {e}", exc_info=True)

    logger.info("Training completed.")
    logger.info(f"Best overall model achieved {best_performance:.2f}% accuracy.")

# ============================
# SCRIPT RUNNER
# ============================
def run_script(script_name):
    try:
        logger.info(f"Running script: {script_name}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, script_name)

        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return

        subprocess.run([sys.executable, script_path])

    except Exception as e:
        logger.error(f"Failed to run {script_name}: {e}", exc_info=True)

# ============================
# MENU DISPLAY FUNCTION
# ============================
def print_menu():
    print("""
    ==============================
      SLAI-v.1.5 Main Launcher Menu
    ==============================

    Select a module to run:

    1 - Evolutionary Agent (Current main.py logic)
    2 - Basic RL Agent (CartPole DQN)                       --> main_cartpole.py
    3 - Evolutionary DQN Agent                              --> main_cartpole_evolve.py
    4 - Multi-Task RL Agent                                 --> main_multitask.py
    5 - Meta-Learning Agent (MAML)                          --> main_maml.py
    6 - Recursive Self-Improvement (RSI)                    --> main_rsi.py
    7 - RL Agent                                            --> main_autotune.py
    8 - Safe AI Agent                                       --> main_safe_ai.py
    9 - Collaborative Agents (Task Routing, Shared Memory)  --> collaborative.main_collaborative.py

    0 - Exit
    """)

# ============================
# MAIN FUNCTION
# ============================
def main():
    logger.info("Welcome to SLAI - Self-Learning Autonomous Intelligence")

    while True:
        print_menu()

        choice = input("Enter choice (0-9): ").strip()

        if choice == "1":
            evolutionary_agent_run()

        elif choice == "2":
            run_script("main_cartpole.py")

        elif choice == "3":
            run_script("main_cartpole_evolve.py")

        elif choice == "4":
            run_script("main_multitask.py")

        elif choice == "5":
            run_script("main_maml.py")

        elif choice == "6":
            run_script("main_rsi.py")

        elif choice == "7":
            run_script("main_autotune.py")

        elif choice == "8":
            run_script("main_safe_ai.py")

        elif choice == "9":
            run_script("main_collaborative.py")

        elif choice == "0":
            logger.info("Exiting SLAI launcher. Goodbye!")
            sys.exit(0)

        else:
            logger.warning("Invalid choice. Please enter a number between 0-9.")

# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    main()
