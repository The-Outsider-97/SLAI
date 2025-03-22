import os
import sys
import logging
from utils.logger import setup_logger
import yaml
import torch
from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
from torch.utils.data import DataLoader, TensorDataset

logger = setup_logger('SLAI', level=logging.DEBUG)

# Load config globally (shared)
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def generate_dummy_data(num_samples=500, input_size=10, output_size=2):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


def evolutionary_agent_run():
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
            logger.error(f"Evolution failed in Generation {generation + 1}: {str(e)}", exc_info=True)

    logger.info("Running manual experiment on custom model...")

    try:
        manual_model = agent.build_model(hidden_size=64)
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
        logger.error(f"Manual experiment failed: {str(e)}", exc_info=True)

    logger.info("Training completed.")
    logger.info(f"Best overall model achieved {best_performance:.2f}% accuracy.")


def run_script(script_name):
    try:
        logger.info(f"Running script: {script_name}")
        os.system(f"{sys.executable} {script_name}")
    except Exception as e:
        logger.error(f"Failed to run {script_name}: {str(e)}")


def main():
    logger.info("Welcome to SLAI - Self-Learning Autonomous Intelligence")
    print("""
    ==============================
      SLAI Main Launcher Menu
    ==============================

    Select a module to run:

    1 - Evolutionary Agent (Current main.py logic)
    2 - Basic RL Agent (CartPole DQN)           --> main_cartpole.py
    3 - Evolutionary DQN Agent                  --> main_cartpole_evolve.py
    4 - Multi-Task RL Agent                     --> main_multitask.py
    5 - Meta-Learning Agent (MAML)              --> main_maml.py
    6 - Recursive Self-Improvement (RSI)        --> main_rsi.py

    0 - Exit
    """)

    choice = input("Enter choice (0-6): ").strip()

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

    elif choice == "0":
        logger.info("Exiting SLAI launcher. Goodbye!")
        sys.exit(0)

    else:
        logger.warning("Invalid choice. Please enter a number between 0-6.")
        main()  # Re-run menu if invalid choice


if __name__ == "__main__":
    main()
