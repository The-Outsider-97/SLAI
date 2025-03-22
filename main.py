# main.py

from agents.evolution_agent import EvolutionAgent
from evaluators.performance_evaluator import PerformanceEvaluator
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def generate_dummy_data(num_samples=500, input_size=10, output_size=2):
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)

def main():
    input_size = 10
    output_size = 2

    agent = EvolutionAgent(input_size=input_size, output_size=output_size)
    evaluator = PerformanceEvaluator(threshold=70.0)

    best_performance = 0
    best_model = None

    train_loader = generate_dummy_data()
    val_loader = generate_dummy_data()

    for generation in range(5):
        print(f"\nGeneration {generation+1}")

        model = agent.build_model()
        agent.train_model(model, train_loader, val_loader)
        performance = agent.evaluate_model(model, val_loader)

        print(f"Model accuracy: {performance:.2f}%")

        if evaluator.is_better(performance, best_performance):
            print(f"New best model found! Accuracy: {performance:.2f}%")
            best_performance = performance
            best_model = model

    print("\nTraining completed.")
    print(f"Best model achieved {best_performance:.2f}% accuracy.")

if __name__ == "__main__":
    main()
