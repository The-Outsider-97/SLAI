import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MultiTaskPolicy(nn.Module):
    """
    A Multi-Task Policy Network with Task Embeddings.
    
    Inputs:
        - state (batch_size x state_size): state observation from environment
        - task_id (batch_size x 1): task identifier tensor

    Outputs:
        - action probabilities for the environment's action space
    """

    def __init__(self, state_size, action_size, task_embedding_size=16, num_tasks=10, hidden_size=128):
        """
        Args:
            state_size (int): Dimension of the state vector.
            action_size (int): Number of discrete actions.
            task_embedding_size (int): Size of the task embedding vector.
            num_tasks (int): Number of different tasks to embed.
            hidden_size (int): Hidden layer size for shared layers.
        """
        super(MultiTaskPolicy, self).__init__()

        # ===============================
        # Embedding for Task ID
        # ===============================
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_size)

        # ===============================
        # Input: state + task embedding --> shared layers
        # ===============================
        self.fc1 = nn.Linear(state_size + task_embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # ===============================
        # Model Info
        # ===============================
        self.state_size = state_size
        self.action_size = action_size
        self.task_embedding_size = task_embedding_size
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size

    def forward(self, state, task_id):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): Tensor of shape (batch_size x state_size)
            task_id (torch.Tensor): Tensor of shape (batch_size) containing task IDs
        
        Returns:
            action_probs (torch.Tensor): Tensor of shape (batch_size x action_size)
        """
        # ===============================
        # Validate Inputs
        # ===============================
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        if task_id.ndim == 0:
            task_id = task_id.unsqueeze(0)  # (1,)

        if torch.any(task_id >= self.num_tasks):
            raise ValueError(f"task_id contains invalid index. Max allowed: {self.num_tasks - 1}")

        # ===============================
        # Embedding and Concatenation
        # ===============================
        task_embed = self.task_embedding(task_id)  # (batch_size, task_embedding_size)

        # Concatenate state and task embedding
        x = torch.cat([state, task_embed], dim=-1)

        # ===============================
        # Shared Fully Connected Layers
        # ===============================
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        # ===============================
        # Output Probabilities (Softmax)
        # ===============================
        action_probs = F.softmax(logits, dim=-1)

        return action_probs
