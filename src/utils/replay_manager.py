
import yaml
import numpy as np

from src.utils.buffer.replay_buffer import ReplayBuffer
from src.utils.buffer.distributed_replay_buffer import DistributedReplayBuffer
from logs.logger import get_logger

logger = get_logger("Replay Manager")

CONFIG_PATH = "src/utils/buffer/configs/buffer_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config: base_config.update(user_config)
    return base_config

class ReplayManager:
    def __init__(self, policy_net, user_config=None):
        """Initialize with config-based buffer selection"""
        config = get_merged_config(user_config)
        mgr_config = config.get('manager', {})
        
        # Initialize appropriate buffer
        if 'distributed' in config:
            self.buffer = DistributedReplayBuffer(user_config)
            logger.info("Using distributed replay buffer")
        else:
            rb_config = config.get('replay_buffer', {})
            self.buffer = ReplayBuffer(rb_config.get('capacity', 100000))
            logger.info("Using basic replay buffer")

        self.policy_net = policy_net
        self.batch_size = mgr_config.get('batch_size', 32)
        self.per_beta = mgr_config.get('per_beta', 0.4)
        self.per_epsilon = mgr_config.get('per_epsilon', 0.01)

        logger.info(f"Initialized Replay Manager with: {self.buffer})")

    def process_batch(self):
        """Main training loop integration"""
        if len(self.buffer) < self.batch_size:
            return None

        # Handle different buffer types
        if isinstance(self.buffer, DistributedReplayBuffer):
            batch, indices, weights = self.buffer.sample(
                self.batch_size,
                strategy='prioritized',
                beta=self.per_beta
            )
        else:
            batch = self.buffer.sample(self.batch_size)
            weights = np.ones(self.batch_size)
            indices = []

        # Training logic
        losses = []
        states, actions, rewards, next_states, dones = batch
        
        for idx, (state, action, reward, next_state) in enumerate(zip(states, actions, rewards, next_states)):
            td_error = self._update_policy(state, action, reward, next_state)
            losses.append(td_error)
            
            if isinstance(self.buffer, DistributedReplayBuffer):
                new_priority = abs(td_error) + self.per_epsilon
                self.buffer.update_priorities([indices[idx]], [new_priority])

        logger.info(f"Training complete. Avg loss: {np.mean(losses):.4f}")
        return np.mean(losses)

    def _update_policy(self, state, action, reward, next_state):
        """Dummy implementation - should be replaced with actual network update"""
        return np.random.rand()  # Return random TD error for demonstration

    def add_experience(self, agent_id, state, action, reward, next_state, done):
        """Universal experience addition interface"""
        if isinstance(self.buffer, DistributedReplayBuffer):
            self.buffer.push(agent_id, state, action, reward, next_state, done)
        else:
            self.buffer.push((state, action, reward, next_state, done))

    def get_buffer_stats(self):
        """Get common buffer statistics"""
        return {
            'size': len(self.buffer),
            'avg_reward': np.mean([exp[3] for exp in self.buffer]),
            'capacity': self.buffer.capacity if hasattr(self.buffer, 'capacity') else self.buffer.maxlen
        }

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Replay Manager ===\n")
    
    # Example policy network (dummy implementation)
    class PolicyNet:
        def update(self, state, action, reward, next_state):
            return np.random.rand()

    manager = ReplayManager(PolicyNet())
    
    # Add dummy experiences
    for i in range(100):
        manager.add_experience(
            agent_id=f"agent_{i%3}",
            state=np.random.rand(4),
            action=np.random.randint(0, 2),
            reward=np.random.rand(),
            next_state=np.random.rand(4),
            done=False
        )
    
    # Train with experiences
    avg_loss = manager.process_batch()
    print(f"\nTraining result - Average loss: {avg_loss:.4f}")
    
    print("\n=== Successfully Ran Replay Manager ===\n")
