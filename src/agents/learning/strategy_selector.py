
import torch
import numpy as np
import random
from collections import deque
from torch import nn

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.learning.learning_calculations import LearningCalculations
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Strategy Selector")
printer = PrettyPrinter

class StrategySelector:
    def __init__(self, config, agent_strategies_map, state_embedder, policy_net, optimizer, loss_fn, device):
        self.config = load_global_config()
        self.embedding_buffer = deque(maxlen=config.get('embedding_buffer_size'))

        self.strategy_config = get_config_section('strategy_selector')
        self.task_embedding_dim = self.strategy_config.get('task_embedding_dim')
        self.min_batch = self.strategy_config.get('meta_controller_batch_size')

        self.agent_strategies_map = agent_strategies_map
        self.state_embedder = state_embedder
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def generate_task_embedding(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Handle 1D vs 2D tensors
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        # Pad/truncate to 256 features
        n, d = state.shape
        if d < 256:
            padding = torch.zeros(n, 256 - d, dtype=state.dtype)
            state = torch.cat([state, padding], dim=1)
        else:
            state = state[:, :256]
        
        return self.state_embedder(state).squeeze(0)

    def observe(self, task_embedding: torch.Tensor, best_agent_strategy_name: str):
        """
        Stores a (task_embedding, best_agent_label) pair for training the meta-controller.
        Robust validation and logging included.

        Args:
            task_embedding (torch.Tensor): Processed task representation
            best_agent_strategy_name (str): Name of best-performing strategy
        """
        # Input validation
        if task_embedding is None or best_agent_strategy_name is None:
            logger.warning("observe() called with None arguments")
            return
            
        if not isinstance(task_embedding, torch.Tensor):
            logger.error(f"task_embedding must be Tensor, got {type(task_embedding)}")
            return
            
        if best_agent_strategy_name not in self.agent_strategies_map:
            valid_strategies = list(self.agent_strategies_map.keys())
            logger.error(f"Invalid strategy '{best_agent_strategy_name}'. Valid: {valid_strategies}")
            return

        # Create label tensor
        label = self.agent_strategies_map[best_agent_strategy_name]
        label_tensor = torch.tensor([label], dtype=torch.long)
        
        # Store in buffer with cloning to prevent reference issues
        self.embedding_buffer.append((
            task_embedding.clone().detach(),
            label_tensor.clone().detach()
        ))
        
        logger.debug(f"Buffered embedding for strategy: {best_agent_strategy_name} "
                     f"(Label: {label}, Buffer size: {len(self.embedding_buffer)})")

    def train_from_embeddings(self):
        """Train meta-controller using buffered embeddings"""
        self.learning_calculations = LearningCalculations()
        
        if len(self.embedding_buffer) < self.min_batch:
            logger.info(f"Deferring training: {len(self.embedding_buffer)}/{self.min_batch} samples")
            return

        # Prepare batch
        embeddings, labels = zip(*random.sample(self.embedding_buffer, self.min_batch))
        embeddings = torch.stack(embeddings)
        labels = torch.cat(labels)

        device = next(self.policy_net.parameters()).device
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        # Training step
        self.policy_net.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.policy_net(embeddings)
        
        # Loss calculation
        loss = self.loss_fn(logits, labels)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Prevent explosions
        
        # Logging
        logger.info(f"Meta-controller training | Loss: {loss.item():.4f} | "
                    f"Accuracy: {self.learning_calculations._calculate_accuracy(logits, labels):.2%}")
        
        # Clear buffer after successful training
        self.embedding_buffer.clear()

    def select_strategy(self, state_embedding):
        if state_embedding.ndim == 1:
            state_embedding = state_embedding.unsqueeze(0)
        
        # Pad/truncate to expected dimension
        expected_dim = self.task_embedding_dim
        actual_dim = state_embedding.shape[1]
        if actual_dim < expected_dim:
            padding = torch.zeros((state_embedding.shape[0], expected_dim - actual_dim), 
                                 dtype=state_embedding.dtype)
            state_embedding = torch.cat([state_embedding, padding], dim=1)
        elif actual_dim > expected_dim:
            state_embedding = state_embedding[:, :expected_dim]
            
        state_embedding = state_embedding.to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(state_embedding)
            probs = torch.softmax(logits.squeeze(0), dim=0)
            strategy_index = torch.argmax(probs).item()
        
        strategy_names = list(self.agent_strategies_map.keys())
        return strategy_names[strategy_index]