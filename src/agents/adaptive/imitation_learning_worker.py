import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from collections import deque
from typing import List, Dict, Callable, Optional, Tuple, Any

from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Imitation Learning")
printer = PrettyPrinter

class ImitationLearningWorker:
    """
    Imitation Learning Worker with Behavior Cloning and DAgger
    - Learns from demonstration data
    - Supports online aggregation of expert demonstrations
    - Integrates with RL policies through mixed objectives
    
    Features:
    - Pure behavior cloning for initial training
    - DAgger for ongoing interactive learning
    - Uncertainty-aware sampling for expert queries
    - Seamless integration with RL policies
    """
    def __init__(self, action_dim: int, state_dim: int, policy_network: nn.Module):
        """
        Initialize Imitation Learning Worker
        
        Args:
            action_dim: Dimensionality of action space
            state_dim: Dimensionality of state space
            policy_network: Reference to the policy network to train
        """
        self.config = load_global_config()
        self.batch_size = self.config.get('batch_size')

        self.il_config = get_config_section('imitation_learning')
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.policy_net = policy_network
        
        # Learning parameters
        self.lr = self.il_config.get('learning_rate', 0.001)
        self.clip_value = self.il_config.get('grad_clip', 1.0)
        self.mix_ratio = self.il_config.get('rl_mix_ratio', 0.7)
        self.entropy_threshold = self.il_config.get('entropy_threshold', 0.5)

        self.memory = MultiModalMemory()
        
        # Optimization
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.lr
        )
        self.loss_fn = nn.MSELoss() if self.il_config.get('continuous_actions', False) \
                       else nn.CrossEntropyLoss()
        
        # Demonstration memory
        self.demo_memory = deque(maxlen=self.il_config.get('demo_capacity', 10000))
        self.dagger_memory = deque(maxlen=self.il_config.get('dagger_capacity', 5000))

        # DAgger parameters
        self.dagger_frequency = self.il_config.get('dagger_frequency', 5)
        self.query_prob = self.il_config.get('initial_query_prob', 0.8)
        self.query_decay = self.il_config.get('query_decay', 0.99)
        
        # Expert interface
        self.expert_policy = None
        self.update_count = 0
        self.device = next(self.policy_net.parameters()).device

        # Demonstration persistence path
        adaptive_root = Path(__file__).resolve().parent
        self.demo_dir = adaptive_root / "demo"
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Imitation Learning Worker initialized")

    def get_action(self, state: np.ndarray) -> int:
        """Get action from imitation learning policy"""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.policy_net(state_tensor)
            if self.il_config.get('continuous_actions', False):
                return output.squeeze().numpy()
            else:
                return torch.argmax(output, dim=-1).item()
    
    def register_expert(self, expert: Callable[[np.ndarray], np.ndarray]):
        """Register expert policy function"""
        self.expert_policy = expert
        logger.info("Expert policy registered")
    
    def load_demonstrations(self, demonstrations: List[Dict]):
        """
        Load offline demonstration dataset
        
        Args:
            demonstrations: List of dicts with keys:
                'state': np.ndarray
                'action': np.ndarray or int
        """
        for demo in demonstrations:
            self.demo_memory.append(self._build_demo_pair(demo['state'], demo['action']))
        logger.info(f"Loaded {len(demonstrations)} offline demonstrations")

    def add_demonstration(self, state: np.ndarray, action: np.ndarray):
        """Add single demonstration sample"""
        self.memory.store_experience(
            state=state,
            action=action,
            reward=2.0,  # Positive reward for demonstrations
            next_state=None,
            done=True,
            context={"source": "demonstration"}
        )

        self.demo_memory.append(self._build_demo_pair(state, action))

    def _build_demo_pair(self, state: np.ndarray, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a consistently-typed (state, action) pair for imitation learning."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if self.il_config.get('continuous_actions'):
            action_tensor = torch.as_tensor(action, dtype=torch.float32)
        else:
            action_tensor = torch.as_tensor(action, dtype=torch.long).reshape(-1)[0]
        return state_tensor, action_tensor

    def _sample_batch(self, dataset: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample a training batch from a dataset."""
        sample_size = min(batch_size, len(dataset))
        batch = random.sample(dataset, sample_size)
        states, actions = zip(*batch)
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        return states, actions
    
    def behavior_cloning(self, epochs: int = 10):
        """Train using pure behavior cloning"""
        if len(self.demo_memory) < self.batch_size:
            logger.warning("Insufficient demonstrations for BC training")
            return
        
            
        dataset = list(self.demo_memory)
        logger.info(f"Starting Behavior Cloning with {len(dataset)} samples")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            random.shuffle(dataset)
            num_batches = 0
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i+self.batch_size]
                states, actions = zip(*batch)
                states = torch.stack(states).to(self.device)
                actions = torch.stack(actions).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.policy_net(states)
                
                # Calculate loss
                if self.il_config.get('continuous_actions'):
                    loss = self.loss_fn(outputs, actions)
                else:
                    loss = F.cross_entropy(outputs, actions)
                
                # Backpropagate
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(1, num_batches)
            logger.info(f"BC Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    def dagger_query(self, state: np.ndarray, agent_action: np.ndarray) -> np.ndarray:
        """
        DAgger interaction - decide whether to query expert and update memory
        
        Returns:
            action: Either expert action or agent action
        """
        if self.expert_policy is None:
            return agent_action
            
        # Decide whether to query expert
        should_query = np.random.rand() < self.query_prob
        
        if should_query:
            expert_action = self.expert_policy(state)
            self.dagger_memory.append(self._build_demo_pair(state, expert_action))
            self.query_prob *= self.query_decay  # Reduce query probability
            return expert_action
        
        return agent_action
    
    def uncertainty_query(self, state: np.ndarray) -> bool:
        """
        Uncertainty-based expert query decision
        Uses policy entropy to determine when to query expert
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = F.softmax(self.policy_net(state_tensor), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).item()
            
        return entropy > self.entropy_threshold
    
    def dagger_update(self):
        """Perform DAgger update using aggregated demonstrations"""
        if len(self.dagger_memory) < self.batch_size:
            return
            
        # Combine demo and DAgger memories
        combined_data = list(self.demo_memory) + list(self.dagger_memory)
        states, actions = self._sample_batch(combined_data, self.batch_size)
        
        # Training step
        self.optimizer.zero_grad()
        outputs = self.policy_net(states)
        
        if self.il_config.get('continuous_actions'):
            loss = self.loss_fn(outputs, actions)
        else:
            loss = F.cross_entropy(outputs, actions)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.optimizer.step()
        
        logger.info(f"DAgger Update | Loss: {loss.item():.4f} | Samples: {len(combined_data)}")
        self.update_count += 1

    def mixed_objective_update(self, states: torch.Tensor, actions: torch.Tensor, 
                              advantages: torch.Tensor, rl_loss: torch.Tensor):
        """
        Combined RL and imitation learning update
        
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Advantage estimates
            rl_loss: Computed RL policy loss
        """
        # Imitation loss
        policy_output = self.policy_net(states)
        
        if self.il_config.get('continuous_actions'):
            il_loss = self.loss_fn(policy_output, actions)
        else:
            il_loss = F.cross_entropy(policy_output, actions)
        
        # Combined loss
        combined_loss = (self.mix_ratio * rl_loss) + ((1 - self.mix_ratio) * il_loss)
        
        # Update
        self.optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        self.optimizer.step()
        
        return combined_loss.item()
    
    def get_demonstration_count(self) -> Dict[str, int]:
        """Get current demonstration statistics"""
        return {
            'offline_demos': len(self.demo_memory),
            'dagger_demos': len(self.dagger_memory),
            'total_demos': len(self.demo_memory) + len(self.dagger_memory)
        }

    def save_demonstrations(self, filepath: str):
        """Save demonstrations to file (defaults to src/agents/adaptive/demo/ when relative)."""
        target_path = Path(filepath)
        if not target_path.is_absolute():
            target_path = self.demo_dir / target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "metadata": {
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "continuous_actions": bool(self.il_config.get('continuous_actions', False)),
                "offline_count": len(self.demo_memory),
                "dagger_count": len(self.dagger_memory),
            },
            "offline": [
                {"state": state.cpu().numpy(), "action": action.cpu().numpy()}
                for state, action in self.demo_memory
            ],
            "dagger": [
                {"state": state.cpu().numpy(), "action": action.cpu().numpy()}
                for state, action in self.dagger_memory
            ],
        }
        torch.save(payload, target_path)
        logger.info(f"Saved demonstrations to {target_path}")
    
    def load_demonstrations_from_file(self, filepath: str):
        """Load demonstrations from file."""
        source_path = Path(filepath)
        if not source_path.is_absolute():
            source_path = self.demo_dir / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Demonstration file not found: {source_path}")

        data = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "offline" not in data:
            raise ValueError(f"Unsupported demonstration format in {source_path}")

        self.demo_memory.clear()
        self.dagger_memory.clear()

        for sample in data.get("offline", []):
            self.demo_memory.append(self._build_demo_pair(sample["state"], sample["action"]))
        for sample in data.get("dagger", []):
            self.dagger_memory.append(self._build_demo_pair(sample["state"], sample["action"]))

        logger.info(
            "Loaded demonstrations from %s | offline=%d dagger=%d",
            source_path,
            len(self.demo_memory),
            len(self.dagger_memory)
        )

if __name__ == "__main__":
    print("\n=== Running Imitation Learning Worker ===\n")
    printer.status("TEST", "Starting Imitation Learning Worker tests", "info")
    from src.agents.learning.utils.policy_network import PolicyNetwork

    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)

    state_dim = 10   # match the config
    action_dim = 2   # match the config
    policy_net = PolicyNetwork(state_dim, action_dim)

    worker = ImitationLearningWorker(
        action_dim=action_dim,
        state_dim=state_dim,
        policy_network=policy_net
    )

    # Dummy expert
    def dummy_expert(state):
        return 0  # always choose action 0

    worker.register_expert(dummy_expert)
    assert worker.get_demonstration_count()['total_demos'] == 0, "Demo memory must start empty."

    # Load demos
    demos = [{'state': np.random.randn(state_dim), 'action': np.random.randint(0, action_dim)} for _ in range(128)]
    worker.load_demonstrations(demos)
    worker.add_demonstration(np.random.randn(state_dim), 1)
    counts = worker.get_demonstration_count()
    assert counts['offline_demos'] == 129, f"Expected 129 demos, got {counts['offline_demos']}"

    # Behavior cloning
    worker.behavior_cloning(epochs=3)
    test_state = np.random.randn(state_dim)
    action = worker.get_action(test_state)
    assert isinstance(action, int), "Expected discrete action output."
    assert 0 <= action < action_dim, "Action index out of range."

    # DAgger query and update
    previous_query_prob = worker.query_prob
    returned_action = worker.dagger_query(test_state, agent_action=1)
    assert returned_action in [0, 1], "DAgger query returned invalid action."
    assert worker.query_prob <= previous_query_prob, "Query probability should decay after query."
    for _ in range(worker.batch_size + 5):
        s = np.random.randn(state_dim)
        worker.dagger_query(s, agent_action=np.random.randint(0, action_dim))
    previous_updates = worker.update_count
    worker.dagger_update()
    assert worker.update_count >= previous_updates, "DAgger update counter did not progress."

    # Uncertainty query
    query_flag = worker.uncertainty_query(np.random.randn(state_dim))
    assert isinstance(query_flag, bool), "Uncertainty query must return bool."

    # Mixed objective update
    states = torch.randn(worker.batch_size, state_dim)
    actions = torch.randint(0, action_dim, (worker.batch_size,), dtype=torch.long)
    advantages = torch.randn(worker.batch_size)
    rl_loss = torch.tensor(0.5, requires_grad=True)
    combined_loss = worker.mixed_objective_update(states, actions, advantages, rl_loss)
    assert isinstance(combined_loss, float), "Mixed objective must return scalar float loss."

    # Save/load path
    save_file = "test_demos.pt"
    worker.save_demonstrations(save_file)
    restore_worker = ImitationLearningWorker(
        action_dim=action_dim,
        state_dim=state_dim,
        policy_network=PolicyNetwork(state_dim, action_dim)
    )
    restore_worker.load_demonstrations_from_file(save_file)
    restored_counts = restore_worker.get_demonstration_count()
    assert restored_counts['offline_demos'] == worker.get_demonstration_count()['offline_demos'], "Offline demo count mismatch after restore."
    assert restored_counts['dagger_demos'] == worker.get_demonstration_count()['dagger_demos'], "DAgger demo count mismatch after restore."

    print("All checks passed.")

    print("\nAll tests completed successfully!\n")
