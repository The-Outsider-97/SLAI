
import torch
import numpy as np
import random

from collections import deque
from torch import nn

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.learning_calculations import LearningCalculations
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Strategy Selector")
printer = PrettyPrinter

class StrategySelector:
    def __init__(self):
        self.config = load_global_config()
        buffer_size = self.config.get('embedding_buffer_size')
        self.embedding_buffer = deque(maxlen=buffer_size)

        self.strategy_config = get_config_section('strategy_selector')
        self.task_embedding_dim = self.strategy_config.get('task_embedding_dim')
        self.min_batch = self.strategy_config.get('min_batch')

        self.calculations = LearningCalculations()

        self.agent_strategies_map = None
        self.state_embedder = None
        self.policy_net = None
        self.optimizer = None
        self.loss_fn = None
        self.device = None

    # ---------- Dependency injection ----------
    def set_agent_strategies_map(self, agent_strategies_map: dict):
        """Set mapping from strategy name to label index."""
        self.agent_strategies_map = agent_strategies_map

    def set_state_embedder(self, state_embedder: nn.Module):
        """Set the neural network that produces task embeddings."""
        self.state_embedder = state_embedder

    def set_policy_network(self, policy_net: nn.Module, optimizer: torch.optim.Optimizer,
                           loss_fn: nn.Module, device: torch.device):
        """Set the meta-controller policy network and its training components."""
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    # ---------- Core methods ----------
    def generate_task_embedding(self, state: np.ndarray) -> torch.Tensor:
        """Generate task embedding from state vector using the state embedder."""
        if self.state_embedder is None:
            raise RuntimeError("State embedder not set. Call set_state_embedder() first.")

        state_tensor = torch.tensor(state, dtype=torch.float32)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Pad/truncate to 256 features (adjust if your embedder expects a different size)
        n, d = state_tensor.shape
        if d < 256:
            padding = torch.zeros(n, 256 - d, dtype=state_tensor.dtype)
            state_tensor = torch.cat([state_tensor, padding], dim=1)
        else:
            state_tensor = state_tensor[:, :256]

        return self.state_embedder(state_tensor).squeeze(0)

    def observe(self, task_embedding: torch.Tensor, best_agent_strategy_name: str):
        """
        Store a (task_embedding, best_agent_label) pair for training the meta-controller.
        """
        if self.agent_strategies_map is None:
            raise RuntimeError("Agent strategies map not set. Call set_agent_strategies_map() first.")

        # Input validation
        if task_embedding is None or best_agent_strategy_name is None:
            logger.warning("observe() called with None arguments")
            return

        if isinstance(task_embedding, np.ndarray):
            task_embedding = torch.tensor(task_embedding, dtype=torch.float32)
        elif isinstance(task_embedding, (list, tuple)):
            task_embedding = torch.tensor(task_embedding, dtype=torch.float32)
        elif not isinstance(task_embedding, torch.Tensor):
            logger.error(f"task_embedding must be Tensor, got {type(task_embedding)}")
            return

        if best_agent_strategy_name not in self.agent_strategies_map:
            valid_strategies = list(self.agent_strategies_map.keys())
            logger.error(f"Invalid strategy '{best_agent_strategy_name}'. Valid: {valid_strategies}")
            return

        # Create label tensor
        label = self.agent_strategies_map[best_agent_strategy_name]
        label_tensor = torch.tensor([label], dtype=torch.long)

        # Store in buffer
        self.embedding_buffer.append((
            task_embedding.clone().detach(),
            label_tensor.clone().detach()
        ))

        logger.debug(f"Buffered embedding for strategy: {best_agent_strategy_name} "
                     f"(Label: {label}, Buffer size: {len(self.embedding_buffer)})")

    def train_from_embeddings(self):
        """Train the meta-controller using buffered embeddings."""
        if self.policy_net is None or self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Policy network components not set. Call set_policy_network() first.")

        if len(self.embedding_buffer) < self.min_batch:
            logger.info(f"Deferring training: {len(self.embedding_buffer)}/{self.min_batch} samples")
            return

        # Sample a batch
        embeddings, labels = zip(*random.sample(self.embedding_buffer, self.min_batch))
        embeddings = torch.stack(embeddings)
        labels = torch.cat(labels)

        device = next(self.policy_net.parameters()).device
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        # Training step
        self.policy_net.train()
        self.optimizer.zero_grad()

        logits = self.policy_net(embeddings)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Logging
        acc = self.calculations._calculate_accuracy(logits, labels)
        logger.info(f"Meta-controller training | Loss: {loss.item():.4f} | Accuracy: {acc:.2%}")

        # Clear buffer after successful training
        self.embedding_buffer.clear()

    def select_strategy(self, state_embedding: torch.Tensor) -> str:
        """Select the best strategy for the given state embedding."""
        if self.policy_net is None:
            raise RuntimeError("Policy network not set. Call set_policy_network() first.")
        if self.device is None:
            raise RuntimeError("Device not set. Provide device in set_policy_network().")

        if state_embedding.ndim == 1:
            state_embedding = state_embedding.unsqueeze(0)

        # Pad/truncate to expected dimension
        actual_dim = state_embedding.shape[1]
        if actual_dim < self.task_embedding_dim:
            padding = torch.zeros((state_embedding.shape[0], self.task_embedding_dim - actual_dim),
                                  dtype=state_embedding.dtype)
            state_embedding = torch.cat([state_embedding, padding], dim=1)
        elif actual_dim > self.task_embedding_dim:
            state_embedding = state_embedding[:, :self.task_embedding_dim]

        state_embedding = state_embedding.to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            logits = self.policy_net(state_embedding)
            probs = torch.softmax(logits.squeeze(0), dim=0)
            strategy_index = torch.argmax(probs).item()

        if self.agent_strategies_map is None:
            raise RuntimeError("Agent strategies map not set. Call set_agent_strategies_map() first.")

        strategy_names = list(self.agent_strategies_map.keys())
        return strategy_names[strategy_index]


if __name__ == "__main__":
    print("\n=== Running Strategy Selector ===\n")
    printer.status("TEST", "Starting Strategy Selector tests", "info")
    from src.agents.learning.utils.policy_network import PolicyNetwork, create_policy_optimizer

    selector = StrategySelector()
    print(selector)
    # ---------- 1. Import actual policy network (relative import works if run as module) ----------
    # Since this script is inside src/agents/learning/utils/, we need to import from sibling modules.
    # We'll use absolute imports for clarity, but the test assumes the package structure.

    # ---------- 2. Configuration ----------
    agent_strategies = {
        "dqn": 0,
        "ppo": 1,
        "evolutionary": 2
    }
    num_strategies = len(agent_strategies)

    # Load config to get dimensions
    config = load_global_config()
    strategy_config = get_config_section("strategy_selector")
    task_embedding_dim = strategy_config.get("task_embedding_dim", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 3. Build actual state embedder ----------
    # The state embedder takes a 256-dim input (after padding) and outputs a task_embedding_dim vector.
    # We'll use a simple two-layer network.
    state_embedder = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, task_embedding_dim)
    ).to(device)

    # ---------- 4. Build actual policy network ----------
    policy_net = PolicyNetwork(
        input_dim=task_embedding_dim,
        output_dim=num_strategies,
        hidden_sizes=[128, 64],
        hidden_activation="relu",
        output_activation="softmax",
        use_batch_norm=False,
        dropout_rate=0.0,
        l1_lambda=0.0,
        l2_lambda=0.0001,
        weight_init="auto",
    ).to(device)

    # Create optimizer using the factory (or directly)
    optimizer = create_policy_optimizer(policy_net)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- 5. Instantiate and inject dependencies ----------
    selector = StrategySelector()
    selector.set_agent_strategies_map(agent_strategies)
    selector.set_state_embedder(state_embedder)
    selector.set_policy_network(policy_net, optimizer, loss_fn, device)

    printer.status("TEST", "Dependencies injected successfully (using real PolicyNetwork)", "success")

    # ---------- 6. Test generate_task_embedding ----------
    random_state = np.random.randn(200).astype(np.float32)  # 200-dim, will be padded to 256
    embedding = selector.generate_task_embedding(random_state)
    assert embedding.shape == (task_embedding_dim,), f"Expected embedding dim {task_embedding_dim}, got {embedding.shape}"
    printer.status("TEST", "generate_task_embedding works", "success")

    # ---------- 7. Test observe and buffer ----------
    min_batch = selector.min_batch
    for i in range(min_batch + 10):
        state = np.random.randn(200).astype(np.float32)
        emb = selector.generate_task_embedding(state)
        strategy = np.random.choice(list(agent_strategies.keys()))
        selector.observe(emb, strategy)

    assert len(selector.embedding_buffer) == min_batch + 10
    printer.status("TEST", f"Buffer size = {len(selector.embedding_buffer)} (expected > min_batch)", "success")

    # ---------- 8. Test training ----------
    selector.train_from_embeddings()
    assert len(selector.embedding_buffer) == 0, "Buffer should be cleared after training"
    printer.status("TEST", "train_from_embeddings executed successfully", "success")

    # ---------- 9. Test select_strategy ----------
    test_state = np.random.randn(200).astype(np.float32)
    test_embedding = selector.generate_task_embedding(test_state)
    chosen_strategy = selector.select_strategy(test_embedding)
    assert chosen_strategy in agent_strategies, f"Invalid strategy returned: {chosen_strategy}"
    printer.status("TEST", f"select_strategy returned: {chosen_strategy}", "success")

    # ---------- 10. Edge cases ----------
    selector.train_from_embeddings()  # Should log "Deferring training"
    printer.status("TEST", "Empty buffer training handled gracefully", "success")

    invalid_embedding = torch.randn(task_embedding_dim)
    selector.observe(invalid_embedding, "non_existent_strategy")  # Should log error
    printer.status("TEST", "Invalid strategy name handled", "success")

    print("\n=== All tests completed successfully! ===\n")
