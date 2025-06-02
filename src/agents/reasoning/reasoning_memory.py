import torch
import os
import random
import numpy as np

from threading import Lock
from datetime import datetime
from collections import namedtuple, defaultdict, OrderedDict

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Memory")
printer = PrettyPrinter

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    """Efficient data structure for proportional sampling with O(log n) complexity"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree structure storage
        self.data = np.zeros(capacity, dtype=object)  # Experience storage
        self.size = 0
        self.write_ptr = 0
        self.max_priority = 1.0  # Default priority for new experiences
        
    def _propagate(self, idx, delta):
        """Update parent nodes after a leaf node change"""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)
            
    def _retrieve(self, idx, value):
        """Find leaf index corresponding to a sample value"""
        left = 2 * idx + 1
        right = left + 1
        
        # Base case: we've reached a leaf node
        if left >= len(self.tree):
            return idx
            
        # Traverse left or right child
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])
    
    def total(self):
        """Get sum of all priorities (root value)"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add experience with priority to the tree"""
        idx = self.write_ptr + self.capacity - 1
        
        # Update data pointer
        self.data[self.write_ptr] = data
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Update tree
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)
        return idx - self.capacity + 1  # Return data index
        
    def update(self, data_idx, priority):
        """Update priority for a given experience"""
        tree_idx = data_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)
        if priority > self.max_priority:
            self.max_priority = priority
            
    def sample(self, value):
        """Sample experience by value (0 <= value <= total)"""
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return (data_idx, self.tree[idx], self.data[data_idx])
    
    def __len__(self):
        return self.size

class ReasoningMemory:
    def __init__(self):
        """
        Manages experiences with SumTree-based prioritized experience replay.
        Memory incorporates a cache mechanism with checkpoints and FIFO principles.
        """
        self.reasoning_memory = []
        self.config = load_global_config()
        self.memory_config = get_config_section('reasoning_memory')

        # Set fallback defaults
        self.memory_config.setdefault('max_size', 10000)
        self.memory_config.setdefault('checkpoint_dir', 'src/agents/reasoning/checkpoints')
        self.memory_config.setdefault('checkpoint_freq', 1000)
        self.memory_config.setdefault('auto_save', True)
        self.memory_config.setdefault('alpha', 0.6)
        self.memory_config.setdefault('beta', 0.4)
        self.memory_config.setdefault('epsilon', 1e-5)
        
        self.tag_index = defaultdict(list)
        self.lock = Lock()
        self.access_counter = 0
        
        # Initialize SumTree for proportional sampling
        self.tree = SumTree(self.memory_config['max_size'])
        self.max_priority = 1.0
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.memory_config['checkpoint_dir'], exist_ok=True)
        logger.info("Reasoning Memory initialized with SumTree")

    def size(self):
        return len(self.tree)

    def add(self, experience, priority=None, tag=None):
        """Add experience with cache management"""
        with self.lock:
            if priority is None:
                priority = self.max_priority
            
            # Apply priority exponent
            priority = (priority + self.memory_config['epsilon']) ** self.memory_config['alpha']
            
            # Add to SumTree
            data_idx = self.tree.add(priority, experience)
            
            # Update tagging system
            if tag:
                self.tag_index[tag].append(data_idx)
                
            self.access_counter += 1
            self.max_priority = max(self.max_priority, priority)
            
            # Automatic checkpointing
            if (self.memory_config['auto_save'] and 
                self.access_counter % self.memory_config['checkpoint_freq'] == 0):
                self.save_checkpoint()

    def get(self, key=None, default=None):
        """Retrieve experience by index or get all experiences"""
        with self.lock:
            if key is not None:
                if isinstance(key, int) and 0 <= key < len(self.tree.data):
                    return self.tree.data[key]
                return default
            return [exp for exp in self.tree.data if exp is not None]

    def set(self, key, value):
        """Update experience at specific index"""
        with self.lock:
            if isinstance(key, int) and 0 <= key < len(self.tree.data):
                self.tree.data[key] = value

    def update_priorities(self, indices, priorities):
        """Update priorities for specific experiences"""
        with self.lock:
            for idx, priority in zip(indices, priorities):
                # Apply priority exponent and update
                priority = (priority + self.memory_config['epsilon']) ** self.memory_config['alpha']
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)

    def sample_proportional(self, batch_size):
        """Sample experiences proportional to their priority"""
        with self.lock:
            if len(self.tree) == 0:
                return [], [], []
                
            samples = []
            indices = []
            priorities = []
            segment = self.tree.total() / batch_size
            
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                value = random.uniform(a, b)
                
                idx, priority, data = self.tree.sample(value)
                samples.append(data)
                indices.append(idx)
                priorities.append(priority)
                
            return samples, indices, priorities

    def save_checkpoint(self, name=None):
        """Save memory state to disk"""
        checkpoint_path = os.path.join(
            self.memory_config['checkpoint_dir'],
            name or f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        
        torch.save({
            'tree_data': self.tree.data,
            'tree_structure': self.tree.tree,
            'tree_write_ptr': self.tree.write_ptr,
            'tree_size': self.tree.size,
            'tag_index': dict(self.tag_index),
            'access_counter': self.access_counter,
            'max_priority': self.max_priority,
            'config': self.memory_config
        }, checkpoint_path)
        logger.info(f"Memory checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, path):
        """Load memory state from disk"""
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            except:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            with self.lock:
                self.tree.data = checkpoint['tree_data']
                self.tree.tree = checkpoint['tree_structure']
                self.tree.write_ptr = checkpoint['tree_write_ptr']
                self.tree.size = checkpoint['tree_size']
                self.tag_index = defaultdict(list, checkpoint.get('tag_index', {}))
                self.access_counter = checkpoint['access_counter']
                self.max_priority = checkpoint['max_priority']
                self.memory_config.update(checkpoint.get('config', {}))
            
            logger.info(f"Loaded memory checkpoint from {path}")
            return True
        logger.warning(f"Checkpoint file {path} not found")
        return False

    def clear(self):
        """Reset memory while preserving configuration"""
        with self.lock:
            self.tree = SumTree(self.memory_config['max_size'])
            self.tag_index = defaultdict(list)
            self.access_counter = 0
            self.max_priority = 1.0

    def get_by_type(self, experience_type: str) -> list:
        """Get experiences by type"""
        with self.lock:
            return [exp for exp in self.tree.data 
                    if exp and exp.get('type') == experience_type]

    def get_high_priority(self, threshold: float = 0.8) -> list:
        """Get high-priority experiences"""
        with self.lock:
            indices = []
            experiences = []
            for i in range(len(self.tree)):
                idx = (self.tree.write_ptr - i - 1) % self.capacity
                if self.tree.tree[idx + self.capacity - 1] >= threshold:
                    indices.append(idx)
                    experiences.append(self.tree.data[idx])
            return experiences

    def metrics(self):
        """Get memory system statistics"""
        return {
            'size': len(self.tree),
            'access_counter': self.access_counter,
            'tags': list(self.tag_index.keys()),
            'total_priority': self.tree.total(),
            'max_priority': self.max_priority
        }

if __name__ == "__main__":
    print("\n=== Running Reasoning Memory ===")
    printer.section_header("Reasoning Memory Initialization")
    bach_size = 1000
    name = "Kabul.json"

    memory = ReasoningMemory()

    print(f"Test for {memory}")
    print(f"{memory.sample_proportional(batch_size=bach_size)}")
    print(f"{memory.save_checkpoint(name=name)}")
    print("\n=== Successfully Ran Reasoning Memory ===\n")
