fixed_main_py = """
import os
import sys
import yaml
import torch
import queue
import logging
import threading
import subprocess

from torch.utils.data import DataLoader, TensorDataset
from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
from frontend.main import launch_ui
from pathlib import Path

# Add parent directory to sys.path for relative imports to work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ========== INIT LOGGING SYSTEM ==========
try:
    from logger import get_logger, get_log_queue
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Logging module import failed: {e}")
    sys.exit(1)

logger = setup_logger('SLAI', level=logging.DEBUG)
log_queue = get_log_queue()
metric_queue = queue.Queue()

# ========== FRONTEND UI LAUNCH ==========
try:
    from frontend.main import launch_ui
    threading.Thread(target=launch_ui, args=(log_queue, metric_queue), daemon=True).start()
except ImportError as e:
    logger.warning(f"UI launch skipped due to missing module: {e}")

# ========== CONFIG LOAD ==========
try:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    logger.error("config.yaml not found.")
    sys.exit(1)
except yaml.YAMLError as e:
    logger.error(f"Error parsing config.yaml: {e}")
    sys.exit(1)

# ========== CONFIG VALIDATION ==========
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

# ========== TORCH DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device in use: {device}")

# ========== DATA LOADER ==========
def generate_dummy_data(num_samples=500, input_size=10, output_size=2):
    X = torch.randn(num_samples, input_size, device=device)
    y = torch.randint(0, output_size, (num_samples,), device=device)
    return DataLoader(TensorDataset(X, y), batch_size=16)

# ========== EVOLUTIONARY AGENT ==========
def evolutionary_agent_run():
    from agents.evolution_agent import EvolutionAgent
    from evaluators.performance_evaluator import PerformanceEvaluator

    os.makedirs('logs', exist_ok=True)
    logger.info("Running Evolutionary Agent...")

    agent = EvolutionAgent(
        input_size=config['agent']['input_size'],
        output_size=config['agent']['output_size'],
        config=config['agent']
    )
    evaluator = PerformanceEvaluator(threshold=config['evaluator']['threshold'])

    train_loader = generate_dummy_data(
        config['training']['num_samples'],
        config['agent']['input_size'],
        config['agent']['output_size']
    )
    val_loader = generate_dummy_data(
        config['training']['num_samples'],
        config['agent']['input_size'],
        config['agent']['output_size']
    )

    agent.initialize_population()
    best_performance = 0
    best_model = None

    for gen in range(10):
        logger.info(f"Generation {gen + 1}")
        try:
            agent.evolve_population(evaluator, train_loader, val_loader)
            best_model = agent.population[0]['model']
            best_performance = agent.population[0]['performance']
            torch.save(best_model.state_dict(), f'logs/best_model_gen_{gen+1}.pth')
            logger.info(f"Gen {gen + 1} - Best Performance: {best_performance:.2f}%")
        except Exception as e:
            logger.error(f"Error in generation {gen+1}: {e}", exc_info=True)

    try:
        manual_model = agent.build_model(hidden_size=64).to(device)
        agent.train_model({'model': manual_model, 'learning_rate': 0.001}, train_loader, val_loader)
        manual_performance = agent.evaluate_model({'model': manual_model}, val_loader)

        if evaluator.is_better(manual_performance, best_performance):
            logger.info(f"Manual model outperformed evolved models with {manual_performance:.2f}% accuracy")
            torch.save(manual_model.state_dict(), 'logs/best_manual_model.pth')
    except Exception as e:
        logger.error("Manual model experiment failed", exc_info=True)

    logger.info("Training complete.")

# ========== SCRIPT LAUNCHER ==========
def run_script(script_name):
    try:
        path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.isfile(path):
            logger.error(f"Script not found: {script_name}")
            return
        subprocess.run([sys.executable, path])
    except Exception as e:
        logger.error(f"Failed to run script {script_name}: {e}", exc_info=True)

# ========== MENU ==========
def print_menu():
    print(\"""
    ==============================
      SLAI-v.1.5 Main Launcher Menu
    ==============================

    Select a module to run:

    1 - Evolutionary Agent
    2 - Basic RL Agent                   --> main_cartpole.py
    3 - Evolutionary DQN Agent          --> main_cartpole_evolve.py
    4 - Multi-Task RL Agent             --> main_multitask.py
    5 - Meta-Learning Agent (MAML)      --> main_maml.py
    6 - Recursive Self-Improvement      --> main_rsi.py
    7 - RL Agent                         --> main_autotune.py
    8 - Safe AI Agent                    --> main_safe_ai.py
    9 - Collaborative Agents            --> main_collaborative.py
    0 - Exit
    \""")

# ========== MAIN ==========
def main():
    logger.info("Welcome to SLAI Framework")
    while True:
        print_menu()
        choice = input("Enter choice (0-9): ").strip()
        if choice == "1":
            evolutionary_agent_run()
        elif choice in map(str, range(2, 10)):
            scripts = {
                "2": "main_cartpole.py",
                "3": "main_cartpole_evolve.py",
                "4": "main_multitask.py",
                "5": "main_maml.py",
                "6": "main_rsi.py",
                "7": "main_autotune.py",
                "8": "main_safe_ai.py",
                "9": "main_collaborative.py"
            }
            run_script(scripts[choice])
        elif choice == "0":
            logger.info("Exiting SLAI launcher.")
            break
        else:
            logger.warning("Invalid choice. Please enter a number between 0-9.")

if __name__ == "__main__":
    main()
"""

main_py_path.write_text(fixed_main_py)
main_py_path
