# main.py

from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
import torch
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

    agent = EvolutionAgent(input_size=input_size, output_size=output_size)
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
    logger.info(f"Starting Generation {generation + 1}")

    agent.evolve_population(evaluator, train_loader, val_loader)

    # Get the best model in the current population
    best_model = agent.population[0]['model']
    best_performance = agent.population[0]['performance']

    logger.info(f"Generation {generation + 1} - Best Accuracy: {best_performance:.2f}%")

    torch.save(best_model.state_dict(), f'logs/best_model_gen_{generation + 1}.pth')

        try:
            model = agent.build_model()
            logger.debug(f"Model architecture: {model}")

            agent.train_model(model, train_loader, val_loader)
            logger.info("Model training complete.")

            performance = agent.evaluate_model(model, val_loader)
            logger.info(f"Model accuracy: {performance:.2f}%")

            if evaluator.is_better(performance, best_performance):
                logger.info(f"New best model found! Accuracy: {performance:.2f}%")
                best_performance = performance
                best_model = model
                torch.save(model.state_dict(), f'logs/best_model_gen_{generation + 1}.pth')
                logger.info(f"Saved best model from Generation {generation + 1}")
            else:
                logger.info(f"Model not better than best ({best_performance:.2f}%).")

        except Exception as e:
            logger.error(f"An error occurred during Generation {generation + 1}: {str(e)}", exc_info=True)

    logger.info("Training completed.")
    logger.info(f"Best model achieved {best_performance:.2f}% accuracy.")

if __name__ == "__main__":
    main()
