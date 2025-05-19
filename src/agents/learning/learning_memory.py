
import yaml
import torch
import os

from datetime import datetime

from logs.logger import get_logger

logger = get_logger("Learning Memory")

CONFIG_PATH = "src/agents/learning/configs/learning_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class LearningMemory:
    def __init__(self, config):
        base_config = config or load_config()
        memory_config = base_config.get('learning_memory', {})
        base_config.update(memory_config)

        self.config = base_config
        self.memory = {}
        self.access_counter = 0
        
        logger.info(f"Learning Memory has succesfully initialized")

    def add(self, experience, tag=None):
        """Add experience with cache management"""
        key = f"{tag}_{datetime.now().timestamp()}" if tag else str(self.access_counter)
        self.memory[key] = experience
        self.access_counter += 1
        
        if len(self.memory) >= self.config['max_size']:
            self._evict()
        
        if self.config['auto_save'] and (self.access_counter % self.config['checkpoint_freq'] == 0):
            self.save_checkpoint()

    def _evict(self):
        """Remove items based on eviction policy"""
        if self.config['eviction_policy'] == 'LRU':
            self.memory.popitem(last=False)
        else:  # FIFO
            self.memory.popitem(last=True)

    def get(self, key=None):
        """Retrieve experience with access tracking"""
        if key:
            experience = self.memory.get(key)
            if experience:
                # Move to end for LRU
                self.memory.pop(key)
                self.memory[key] = experience
            return experience
        return list(self.memory.values())

    def set(self, key, value):
        """Explicitly set a key-value pair in memory"""
        self.memory[key] = value

    def save_checkpoint(self, path=None):
        """Save memory state to disk"""
        checkpoint_path = path or os.path.join(
            self.config['checkpoint_dir'],
            f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        torch.save({
            'memory': self.memory,
            'access_counter': self.access_counter,
            'config': self.config
        }, checkpoint_path)
        logger.info(f"Memory checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path):
        """Load memory state from disk"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.memory = checkpoint['memory']
            self.access_counter = checkpoint['access_counter']
            self.config.update(checkpoint.get('config', {}))
            logger.info(f"Loaded memory checkpoint from {path}")
        else:
            logger.warning(f"Checkpoint file {path} not found")

    def clear(self):
        """Reset memory while preserving configuration"""
        self.memory.clear()
        self.access_counter = 0

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Learning Memory ===\n")

    config = load_config()
    experience = None
    memory = LearningMemory(config)
    
    # Store experiences with optional tagging
    memory.add(experience, tag="dqn_transition")
    
    # Automatic cache management and checkpointing
    experiences = memory.get()  # Get all experiences
    
    # Manual checkpoint control
    memory.save_checkpoint("important_memory.pt")
    memory.load_checkpoint("previous_memory.pt")


    print(f"\n{memory}\n")
    print("\n=== Successfully Learning Memory ===\n")
