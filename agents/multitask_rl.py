import torch
import torch.nn as nn
import torch.optim as optim

class MultiTaskPolicy(nn.Module):
    def __init__(self, state_size, action_size, task_embedding_size=16):
        super().__init__()
        self.task_embedding = nn.Embedding(10, task_embedding_size)  # Assume 10 tasks for example
        self.fc1 = nn.Linear(state_size + task_embedding_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state, task_id):
        task_embed = self.task_embedding(task_id)
        x = torch.cat([state, task_embed], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
