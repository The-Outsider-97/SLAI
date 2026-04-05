import torch
import os
import random
import numpy as np
import pickle
import tempfile
import shutil

from threading import RLock
from datetime import datetime
from collections import namedtuple, defaultdict
from typing import List, Tuple, Any, Optional, Union, Callable, Dict

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Learning Memory")
printer = PrettyPrinter

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    """
    Efficient priority sum tree for proportional sampling.
    Complexity: O(log n) for add, update, and sample.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of experiences.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)  # sum tree
        self.data = np.zeros(capacity, dtype=object)            # experience storage
        self.size = 0                                           # current number of items
        self.write_ptr = 0                                      # next insertion index
        self.max_priority = 1.0                                 # highest priority ever seen

    def _propagate(self, idx: int, delta: float) -> None:
        """Iteratively update parent sums after a leaf priority change."""
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index corresponding to a cumulative probability value."""
        while idx < self.capacity - 1:          # not a leaf
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx

    def total(self) -> float:
        """Sum of all priorities (root value)."""
        return self.tree[0]

    def add(self, priority: float, data: Any) -> int:
        """
        Insert or overwrite an experience.
        Returns the data index (0..capacity-1) where the experience is stored.
        """
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        old_priority = self.tree[idx]
        delta = priority - old_priority
        self.tree[idx] = priority
        self._propagate(idx, delta)

        # Update max priority
        if priority > self.max_priority:
            self.max_priority = priority

        # Advance write pointer and update size
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

        return self.write_ptr - 1 if self.write_ptr != 0 else self.capacity - 1

    def update(self, data_idx: int, priority: float) -> None:
        """Change the priority of an existing experience."""
        tree_idx = data_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)
        if priority > self.max_priority:
            self.max_priority = priority

    def sample(self, value: float) -> Tuple[int, float, Any]:
        """
        Map a uniform random value in [0, total) to a leaf.
        Returns (data_idx, priority, experience).
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return data_idx, self.tree[tree_idx], self.data[data_idx]

    def __len__(self) -> int:
        return self.size


class LearningMemory:
    """
    Prioritized Experience Replay with SumTree, automatic checkpointing, and tag management.
    Supports both proportional prioritization (PER) and uniform sampling.
    """

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section('learning_memory')

        # Default values for missing configuration keys
        self.memory_config.setdefault('max_size', 10000)
        self.memory_config.setdefault('checkpoint_dir', 'checkpoints')
        self.memory_config.setdefault('checkpoint_freq', 1000)
        self.memory_config.setdefault('auto_save', True)
        self.memory_config.setdefault('alpha', 0.6)           # priority exponent
        self.memory_config.setdefault('beta', 0.4)           # initial importance-sampling exponent
        self.memory_config.setdefault('epsilon', 1e-5)       # small offset to avoid zero priority
        self.memory_config.setdefault('beta_annealing_steps', 100000)

        # Internal state
        self.tree = SumTree(self.memory_config['max_size'])
        self.tag_index = defaultdict(set)          # tag -> set of data indices
        self.index_to_tags = defaultdict(set)      # data index -> set of tags (for fast removal)
        self.lock = RLock()
        self.access_counter = 0                    # number of add/sample operations (for annealing)
        self.max_priority = 1.0                    # current max priority for new experiences
        self.key_value_store = {}                  # arbitrary metadata storage

        # Importance sampling beta annealing
        self.beta_start = self.memory_config['beta']
        self.beta_end = 1.0
        self.beta_annealing_steps = self.memory_config['beta_annealing_steps']

        logger.info("LearningMemory initialized with SumTree (PER)")

    # ----------------------------------------------------------------------
    # Core experience management
    # ----------------------------------------------------------------------

    def size(self) -> int:
        """Number of stored experiences."""
        with self.lock:
            return len(self.tree)

    def add(self, experience: Any, priority: Optional[float] = None, tag: Optional[str] = None) -> None:
        """
        Add a new experience.
        If priority is None, the current max priority is used (encourages exploration of new data).
        """
        with self.lock:
            if priority is None:
                priority = self.max_priority

            # Apply exponent α and ensure non-zero
            priority = (priority + self.memory_config['epsilon']) ** self.memory_config['alpha']

            # Adding to SumTree may overwrite an existing slot.
            # Get the data index that will be overwritten (if any) BEFORE adding.
            write_ptr = self.tree.write_ptr
            old_data_idx = write_ptr if self.tree.size < self.tree.capacity else None
            # For capacity not yet full, no overwrite. For full, the slot at write_ptr is overwritten.
            if self.tree.size == self.tree.capacity:
                old_data_idx = write_ptr

            # Insert into tree
            data_idx = self.tree.add(priority, experience)

            # If we overwrote an existing experience, remove its tags
            if old_data_idx is not None and old_data_idx in self.index_to_tags:
                for old_tag in self.index_to_tags[old_data_idx]:
                    self.tag_index[old_tag].discard(old_data_idx)
                    if not self.tag_index[old_tag]:
                        del self.tag_index[old_tag]
                del self.index_to_tags[old_data_idx]

            # Attach new tag
            if tag is not None:
                self.tag_index[tag].add(data_idx)
                self.index_to_tags[data_idx].add(tag)

            self.access_counter += 1
            self.max_priority = max(self.max_priority, priority)

            # Auto‑checkpoint if needed
            if self.memory_config.get('auto_save') and (self.access_counter % self.memory_config['checkpoint_freq'] == 0):
                self.save_checkpoint()

    def add_batch(self, experiences: List[Any], tag: Optional[str] = None) -> None:
        """Add multiple experiences (all receive the same tag if provided)."""
        for exp in experiences:
            self.add(exp, tag=tag)

    def sample_proportional(self, batch_size: int) -> Tuple[List[Any], List[int], List[float]]:
        """
        Sample a batch using prioritized experience replay.
        Returns (samples, indices, importance_sampling_weights).
        """
        with self.lock:
            if len(self.tree) == 0:
                return [], [], []

            # Anneal beta
            beta = self.beta_start + (self.beta_end - self.beta_start) * min(
                1.0, self.access_counter / self.beta_annealing_steps
            )

            total_priority = self.tree.total()
            segment = total_priority / batch_size
            samples = []
            indices = []
            weights = []

            for i in range(batch_size):
                # Uniformly sample within segment to reduce bias
                low = segment * i
                high = segment * (i + 1)
                value = random.uniform(low, high)
                idx, priority, data = self.tree.sample(value)

                # Importance sampling weight: (N * P(i))^(-β)  normalized by max weight
                prob = priority / total_priority
                weight = (len(self.tree) * prob) ** (-beta)
                # Normalization step: divide by max weight (approximated by max priority)
                # For stability we use the stored max_priority of the tree.
                max_weight = (len(self.tree) * (self.tree.max_priority / total_priority)) ** (-beta)
                weight = weight / max_weight if max_weight > 0 else 1.0

                samples.append(data)
                indices.append(idx)
                weights.append(weight)

            return samples, indices, weights

    def sample(self, batch_size: int) -> List[Any]:
        """Uniform random sampling (ignores priorities). Useful for debugging or non‑PER."""
        with self.lock:
            if len(self.tree) == 0:
                return []
            # Collect only non‑None experiences (the data array may have None at empty slots)
            valid_indices = [i for i in range(len(self.tree.data)) if self.tree.data[i] is not None]
            if len(valid_indices) == 0:
                return []
            k = min(batch_size, len(valid_indices))
            chosen = random.sample(valid_indices, k)
            return [self.tree.data[i] for i in chosen]

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities of experiences (e.g., after computing TD errors).
        Priorities should be raw absolute TD errors (will be exponentiated with α internally).
        """
        with self.lock:
            for idx, raw_priority in zip(indices, priorities):
                # Clip to avoid negative, add epsilon, exponentiate
                priority = (abs(raw_priority) + self.memory_config['epsilon']) ** self.memory_config['alpha']
                self.tree.update(idx, priority)
                if priority > self.max_priority:
                    self.max_priority = priority

    def update_priorities_from_model(self, indices: List[int], model: torch.nn.Module,
                                     loss_fn: Callable, gamma: float = 0.99,
                                     device: torch.device = torch.device('cpu')) -> None:
        """
        Convenience method: compute TD errors using a model and update priorities.
        Expects that the stored experiences are Transition tuples.
        """
        # Gather transitions
        transitions = [self.tree.data[idx] for idx in indices]
        if not transitions:
            return

        # Prepare batches
        states = torch.stack([t.state for t in transitions]).to(device)
        actions = torch.tensor([t.action for t in transitions], device=device)
        rewards = torch.tensor([t.reward for t in transitions], device=device)
        next_states = torch.stack([t.next_state for t in transitions]).to(device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=device)

        with torch.no_grad():
            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = model(next_states).max(1)[0]
            target = rewards + gamma * next_q_values * (1 - dones)
            td_errors = (target - q_values).abs().cpu().numpy()

        self.update_priorities(indices, td_errors.tolist())

    # ----------------------------------------------------------------------
    # Tag management
    # ----------------------------------------------------------------------

    def get_by_tag(self, tag: str) -> List[Any]:
        """Retrieve all experiences associated with a tag."""
        with self.lock:
            indices = self.tag_index.get(tag, set())
            return [self.tree.data[idx] for idx in indices if self.tree.data[idx] is not None]

    def delete_tag(self, tag: str) -> None:
        """Remove a tag from all experiences and delete the tag entry."""
        with self.lock:
            if tag not in self.tag_index:
                return
            for idx in self.tag_index[tag]:
                self.index_to_tags[idx].discard(tag)
                if not self.index_to_tags[idx]:
                    del self.index_to_tags[idx]
            del self.tag_index[tag]

    # ----------------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------------

    def get_recent_states(self, num_states: int = 100) -> List[Any]:
        """Return the most recent `num_states` state tensors (from newest to oldest)."""
        with self.lock:
            recent = []
            # Walk backward from write_ptr-1 (most recent) through the circular buffer
            start = (self.tree.write_ptr - 1) % self.tree.capacity
            for i in range(self.tree.capacity):
                idx = (start - i) % self.tree.capacity
                exp = self.tree.data[idx]
                if exp is not None and hasattr(exp, 'state'):
                    recent.append(exp.state)
                    if len(recent) >= num_states:
                        break
            return recent

    def get(self, key: Optional[Union[int, str]] = None, default=None):
        """
        Flexible access:
        - If key is int: return experience at that index (or default if empty).
        - If key is str: return value from key-value store.
        - If key is None: return list of all experiences (excluding None slots).
        """
        with self.lock:
            if key is None:
                return [self.tree.data[i] for i in range(self.tree.capacity) if self.tree.data[i] is not None]
            if isinstance(key, int):
                if 0 <= key < self.tree.capacity and self.tree.data[key] is not None:
                    return self.tree.data[key]
                return default
            return self.key_value_store.get(key, default)

    def set(self, key: Union[int, str], value: Any) -> None:
        """
        Store an experience at a specific index (int) or a metadata key (str).
        Use with caution when overwriting indices – tags are not automatically updated.
        """
        with self.lock:
            if isinstance(key, int):
                if 0 <= key < self.tree.capacity:
                    self.tree.data[key] = value
            else:
                self.key_value_store[key] = value

    def clear(self) -> None:
        """Reset memory to empty state (preserves configuration)."""
        with self.lock:
            self.tree = SumTree(self.memory_config['max_size'])
            self.tag_index.clear()
            self.index_to_tags.clear()
            self.access_counter = 0
            self.max_priority = 1.0
            self.key_value_store.clear()

    def metrics(self) -> Dict[str, Any]:
        """Return useful statistics about the memory."""
        with self.lock:
            total = self.tree.total()
            avg_priority = total / len(self.tree) if len(self.tree) > 0 else 0.0
            return {
                'size': len(self.tree),
                'capacity': self.tree.capacity,
                'access_counter': self.access_counter,
                'tags': list(self.tag_index.keys()),
                'total_priority': total,
                'avg_priority': avg_priority,
                'max_priority': self.max_priority,
                'beta_current': self.beta_start + (self.beta_end - self.beta_start) * min(
                    1.0, self.access_counter / self.beta_annealing_steps
                )
            }

    # ----------------------------------------------------------------------
    # Checkpointing (production grade)
    # ----------------------------------------------------------------------

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        with self.lock:
            checkpoint_dir = self.memory_config['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
    
            if path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(checkpoint_dir, f"memory_{timestamp}.pt")
    
            state = {
                'version': 1,
                'tree_data': self.tree.data,
                'tree_structure': self.tree.tree,
                'tree_write_ptr': self.tree.write_ptr,
                'tree_size': self.tree.size,
                'tree_max_priority': self.tree.max_priority,
                'tag_index': {k: list(v) for k, v in self.tag_index.items()},
                'index_to_tags': {k: list(v) for k, v in self.index_to_tags.items()},
                'access_counter': self.access_counter,
                'max_priority': self.max_priority,
                'key_value_store': self.key_value_store,
                'config': self.memory_config
            }
    
            temp_fd, temp_path = tempfile.mkstemp(dir=checkpoint_dir, prefix='.tmp_memory_')
            try:
                with os.fdopen(temp_fd, 'wb') as f:
                    torch.save(state, f, pickle_protocol=2)   # <-- protocol 2 for weights_only=True compatibility
                shutil.move(temp_path, path)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
    
            logger.info(f"Memory checkpoint saved to {path}")
            return path

    def load_checkpoint(self, path: str) -> None:
        """
        Load a previously saved checkpoint.
        Handles both safe (weights_only=True) and legacy (weights_only=False) formats.
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint file {path} not found")
            return

        with self.lock:
            # Try safe loading first
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            except Exception as e:
                logger.warning(f"Safe loading failed for {path}: {e}. Falling back to full loading.")
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)

            # Validate version (future expansion)
            version = checkpoint.get('version', 1)
            if version != 1:
                logger.warning(f"Checkpoint version {version} may not be fully compatible")

            # Restore SumTree
            self.tree.data = checkpoint['tree_data']
            self.tree.tree = checkpoint['tree_structure']
            self.tree.write_ptr = checkpoint['tree_write_ptr']
            self.tree.size = checkpoint['tree_size']
            self.tree.max_priority = checkpoint['tree_max_priority']

            # Restore tags (convert lists back to sets)
            self.tag_index = {k: set(v) for k, v in checkpoint.get('tag_index', {}).items()}
            self.index_to_tags = {k: set(v) for k, v in checkpoint.get('index_to_tags', {}).items()}

            # Restore other state
            self.access_counter = checkpoint['access_counter']
            self.max_priority = checkpoint['max_priority']
            self.key_value_store = checkpoint.get('key_value_store', {})
            # Update config (but keep current checkpoint_dir etc. unless overwritten)
            saved_config = checkpoint.get('config', {})
            self.memory_config.update(saved_config)

            logger.info(f"Loaded memory checkpoint from {path} (version {version})")


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Production Learning Memory ===\n")

    # Dummy transition factory
    def dummy_state():
        return torch.randn(4)

    memory = LearningMemory()

    # Add experiences with tags
    for i in range(2000):
        exp = Transition(dummy_state(), i % 4, 1.0, dummy_state(), False)
        memory.add(exp, tag="main_trajectory")

    # Sample using PER
    batch, indices, weights = memory.sample_proportional(batch_size=64)

    # Simulate model update (replace with real model)
    fake_td_errors = np.random.randn(len(indices)) ** 2
    memory.update_priorities(indices, fake_td_errors)

    # Retrieve by tag
    tagged_exps = memory.get_by_tag("main_trajectory")
    print(f"Found {len(tagged_exps)} experiences with tag 'main_trajectory'")

    # Checkpoint
    ckpt_path = memory.save_checkpoint()
    print(f"Checkpoint saved to {ckpt_path}")

    # Clear and reload
    memory.clear()
    print(f"After clear: size = {memory.size()}")
    memory.load_checkpoint(ckpt_path)
    print(f"After reload: size = {memory.size()}")

    print("\n=== Learning Memory Production Ready ===")
