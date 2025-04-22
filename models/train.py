import numpy as np
from src.agents.perception_agent import PerceptionAgent
from data.multimodal_dataset import MultimodalDataset

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2), 2 * (pred - target) / pred.size

def train(agent, dataset, epochs=10, lr=1e-3):
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataset:
            agent.zero_grad()
            output = agent.forward(batch)
            target = np.random.randn(*output.shape)  # Dummy supervision
            loss, dout = mse_loss(output, target)
            agent.backward(dout)
            agent.step(learning_rate=lr)
            total_loss += loss
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.4f}")
