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
    
    def get_current_context(self) -> list:
        """
        Dynamically generates current context tags based on:
        - Recent high-priority experiences
        - Semantic clustering of recent memories
        - Memory saturation levels
        - Temporal patterns
        Returns a list of context tags describing the current cognitive state
        """
        context_tags = []
        if self.tree.size == 0:
            return ["empty_memory"]
        
        # Configuration parameters
        recent_window = min(50, self.tree.size)
        high_priority_threshold = 0.85
        context_decay_factor = 0.95
        
        # Calculate memory saturation
        saturation = self.tree.size / self.tree.capacity
        if saturation > 0.9:
            context_tags.append("memory_saturated")
        
        # Analyze recent experiences
        recent_indices = self._get_recent_indices(recent_window)
        recent_tags = []
        high_priority_count = 0
        
        for data_idx in recent_indices:
            # Get priority value
            tree_idx = data_idx + self.tree.capacity - 1
            priority = self.tree.tree[tree_idx]
            
            # Check for high-priority items
            if priority > high_priority_threshold:
                high_priority_count += 1
            
            # Get experience and its tags
            exp = self.tree.data[data_idx]
            if exp and 'tag' in exp:
                recent_tags.append(exp['tag'])
        
        # Add priority context
        if high_priority_count > recent_window * 0.3:
            context_tags.append("high_priority_context")
        
        # Add most frequent tags from recent window
        if recent_tags:
            freq_dist = defaultdict(int)
            for tag in recent_tags:
                freq_dist[tag] += 1
            top_tags = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            context_tags.extend([tag for tag, _ in top_tags])
        
        # Add temporal context
        hour = datetime.now().hour
        if 5 <= hour < 12:
            context_tags.append("morning_context")
        elif 18 <= hour < 22:
            context_tags.append("evening_context")
        
        # Apply context decay to historical tags
        if hasattr(self, 'historical_context'):
            decayed_context = []
            for tag, weight in self.historical_context.items():
                new_weight = weight * context_decay_factor
                if new_weight > 0.1:
                    decayed_context.append((tag, new_weight))
            self.historical_context = dict(decayed_context)
        else:
            self.historical_context = {}
        
        # Update historical context
        for tag in set(context_tags):
            self.historical_context[tag] = self.historical_context.get(tag, 0) + 1.0
        
        # Combine with persistent context
        persistent_tags = [tag for tag, weight in self.historical_context.items() 
                          if weight > 0.5]
        return list(set(context_tags + persistent_tags))
    
    def _get_recent_indices(self, count: int) -> list:
        """Get indices of most recent experiences in chronological order"""
        if count <= 0:
            return []
        
        count = min(count, self.tree.size)
        indices = []
        
        if self.tree.write_ptr >= count:
            start = self.tree.write_ptr - count
            indices = list(range(start, self.tree.write_ptr))
        else:
            wrap_count = count - self.tree.write_ptr
            indices = (list(range(self.tree.capacity - wrap_count, self.tree.capacity)) +
                      list(range(0, self.tree.write_ptr)))
        
        return indices

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
