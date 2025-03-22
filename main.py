# main.py

from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset
import yaml
from utils.logger import setup_logger

logger = setup_logger('SLAI', level=logging.DEBUG)

logger.info("Starting SLAI self-improving agent")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def generate_dummy_data(num_samples=500, input_size=10, output_size=2):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)

def main():
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

    # Initialize and evolve population
    agent.initialize_population()

    for generation in range(10):
        logger.info(f"Starting Evolutionary Generation {generation + 1}")

        try:
            # Evolve population automatically
            agent.evolve_population(evaluator, train_loader, val_loader)

            # Get best evolved model
            best_model = agent.population[0]['model']
            best_performance = agent.population[0]['performance']

            logger.info(f"Evolved Best Model - Gen {generation + 1}: {best_performance:.2f}% accuracy")

            # Save best evolved model
            torch.save(best_model.state_dict(), f'logs/best_model_evolved_gen_{generation + 1}.pth')

        except Exception as e:
            logger.error(f"Evolution failed in Generation {generation + 1}: {str(e)}", exc_info=True)

    # OPTIONAL: Run a separate manual experiment (manual build/train/eval)
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

if __name__ == "__main__":
    main()
