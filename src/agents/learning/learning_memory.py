
import torch
import os
import random
import numpy as np

from threading import Lock
from datetime import datetime
from collections import namedtuple, defaultdict, OrderedDict

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("Learning Memory")

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

class LearningMemory:
    def __init__(self):
        """Manages experiences with SumTree-based prioritized experience replay."""
        self.config = load_global_config()
        self.memory_config = get_config_section('learning_memory')

        # Set fallback defaults
        self.memory_config.setdefault('max_size', 10000)
        self.memory_config.setdefault('checkpoint_dir', 'checkpoints')
        self.memory_config.setdefault('checkpoint_freq', 1000)
        self.memory_config.setdefault('auto_save', True)
        self.memory_config.setdefault('alpha', 0.6)  # Priority exponent
        self.memory_config.setdefault('beta', 0.4)   # Importance-sampling exponent
        self.memory_config.setdefault('epsilon', 1e-5)  # Avoid zero priority

        self.tag_index = defaultdict(list)
        self.lock = Lock()
        self.access_counter = 0
        
        # Initialize SumTree for proportional sampling
        self.tree = SumTree(self.memory_config['max_size'])
        self.max_priority = 1.0  # Initial priority for new experiences
        
        # Beta annealing parameters
        self.beta_start = self.memory_config['beta']
        self.beta_end = 1.0
        self.beta_annealing_steps = 100000

        logger.info("Learning Memory successfully initialized with SumTree")

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
            if self.memory_config.get('auto_save') and (self.access_counter % self.memory_config['checkpoint_freq'] == 0):
                self.save_checkpoint()

    def add_batch(self, experiences, tag=None):
        for experience in experiences:
            self.add(experience, tag=tag)

    def sample(self, batch_size):
        """Uniform random sampling"""
        with self.lock:
            if len(self.tree) == 0:
                return []
            keys = random.sample(range(len(self.tree)), min(batch_size, len(self.tree)))
            return [self.tree.data[key] for key in keys]

    def sample_proportional(self, batch_size):
        """Sample experiences proportional to their priority using SumTree"""
        with self.lock:
            if len(self.tree) == 0:
                return [], [], []
                
            samples = []
            indices = []
            priorities = []
            segment = self.tree.total() / batch_size
            beta = self.beta_start + (self.beta_end - self.beta_start) * min(
                1.0, self.access_counter / self.beta_annealing_steps
            )
            
            for i in range(batch_size):
                # Sample within a segment to ensure good coverage
                a = segment * i
                b = segment * (i + 1)
                value = random.uniform(a, b)
                
                # Retrieve sample from SumTree
                idx, priority, data = self.tree.sample(value)
                
                # Calculate importance-sampling weight
                prob = priority / self.tree.total()
                weight = (len(self.tree) * prob) ** (-beta)
                weight /= self.tree.max_priority  # Normalize
                
                samples.append(data)
                indices.append(idx)
                priorities.append(weight)
                
            return samples, indices, priorities
        
    def compute_new_priorities(self, samples):
        """Placeholder for priority calculation (should be implemented by user)"""
        return [1.0] * len(samples)  # Default uniform priority

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        with self.lock:
            for idx, priority in zip(indices, priorities):
                # Apply priority exponent and update
                priority = (priority + self.memory_config['epsilon']) ** self.memory_config['alpha']
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)

    def get_by_tag(self, tag):
        """Get experiences by tag"""
        with self.lock:
            indices = self.tag_index.get(tag, [])
            return [self.tree.data[idx] for idx in indices]

    def get(self, key=None, default=None):
        with self.lock:
            if key is not None:
                # Add type check
                if not isinstance(key, int):
                    return default
                if key < len(self.tree.data) and self.tree.data[key] is not None:
                    return self.tree.data[key]
                return default
            return [self.tree.data[i] for i in range(len(self.tree)) 
                    if self.tree.data[i] is not None]

    def set(self, key, value):
        """Set experience at specific index"""
        with self.lock:
            if key < len(self.tree.data):
                self.tree.data[key] = value

    def save_checkpoint(self, path=None):
        """Save memory state to disk"""
        checkpoint_path = path or os.path.join(
            self.memory_config['checkpoint_dir'],
            f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        torch.save({
            'tree_data': self.tree.data,
            'tree_structure': self.tree.tree,
            'tree_write_ptr': self.tree.write_ptr,
            'tree_size': self.tree.size,
            'tree_max_priority': self.tree.max_priority,
            'tag_index': dict(self.tag_index),
            'access_counter': self.access_counter,
            'max_priority': self.max_priority,
            'config': self.memory_config
        }, checkpoint_path)
        logger.info(f"Memory checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path):
        """Load memory state from disk with compatibility for PyTorch 2.6+ safety features"""
        if os.path.exists(path):
            try:
                # First try with weights_only=True for security
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            except Exception as e:
                # If that fails, try with weights_only=False for compatibility
                logger.warning(f"Safe loading failed, using compatibility mode: {str(e)}")
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Restore SumTree state
            self.tree.data = checkpoint['tree_data']
            self.tree.tree = checkpoint['tree_structure']
            self.tree.write_ptr = checkpoint['tree_write_ptr']
            self.tree.size = checkpoint['tree_size']
            self.tree.max_priority = checkpoint['tree_max_priority']
            
            # Restore other state
            self.tag_index = defaultdict(list, checkpoint.get('tag_index', {}))
            self.access_counter = checkpoint['access_counter']
            self.max_priority = checkpoint['max_priority']
            self.memory_config.update(checkpoint.get('config', {}))
            
            logger.info(f"Loaded memory checkpoint from {path}")
        else:
            logger.warning(f"Checkpoint file {path} not found")

    def metrics(self):
        return {
            'size': len(self.tree),
            'access_counter': self.access_counter,
            'tags': list(self.tag_index.keys()),
            'total_priority': self.tree.total(),
            'max_priority': self.max_priority
        }

    def clear(self):
        """Reset memory while preserving configuration"""
        with self.lock:
            self.tree = SumTree(self.memory_config['max_size'])
            self.tag_index = defaultdict(list)
            self.access_counter = 0
            self.max_priority = 1.0

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Learning Memory ===\n")

    experience = None
    memory = LearningMemory()
    
    # Store experiences with optional tagging
    memory.add(experience, tag="dqn_transition")
    
    # Automatic cache management and checkpointing
    experiences = memory.get()  # Get all experiences
    for experience in experiences:
        memory.add(experience, priority=0.5)

    samples, indices, weights = memory.sample_proportional(batch_size=64)

    new_priorities = memory.compute_new_priorities(samples)
    memory.update_priorities(indices, new_priorities)
    
    # Manual checkpoint control
    memory.save_checkpoint("important_memory.pt")
    memory.load_checkpoint("important_memory.pt")

    print(f"\nMemory Metrics: {memory.metrics()}")
    print("\n=== Successfully Ran Learning Memory ===\n")
