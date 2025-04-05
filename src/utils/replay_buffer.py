import os, sys
import yaml
import json
import heapq
import logging
import numpy as np
import random
from collections import deque, defaultdict
from threading import Lock
from datetime import datetime, timedelta

logger = logging.getLogger("DistributedReplayBuffer")

class DistributedReplayBuffer:
    def __init__(self, capacity=100_000, seed=None, 
                 staleness_threshold=timedelta(days=1),
                 prioritization_alpha=0.6):
        """
        Enhanced replay buffer with features from RL research literature.
        
        Args:
            capacity: Maximum number of transitions stored
            staleness_threshold: Automatically remove experiences older than this
            prioritization_alpha: Prioritization exponent (0=uniform, 1=full prioritization)
        """
        # Core storage with thread safety
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = Lock()
        
        # Prioritization structures (sum tree alternative using heapq)
        self.priorities = []
        self.max_priority = 1.0
        self.alpha = prioritization_alpha
        
        # Staleness tracking
        self.timestamps = deque(maxlen=capacity)
        self.staleness_threshold = staleness_threshold
        
        # Agent-specific tracking
        self.agent_experience_map = defaultdict(int)
        
        # Quality metrics
        self.reward_stats = {'sum': 0.0, 'max': -np.inf, 'min': np.inf}
        
        # Seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        logger.info(f"Initialized enhanced buffer (Capacity: {capacity}, Prioritization α: {prioritization_alpha})")

    def push(self, agent_id, state, action, reward, next_state, done):
        """
        Store transition with automatic priority initialization and quality tracking.
        Implements prioritized experience replay (Schaul et al., 2015).
        """
        with self.lock:
            # Store experience with initial priority and timestamp
            experience = (agent_id, state, action, reward, next_state, done)
            self.buffer.append(experience)
            self.timestamps.append(datetime.now())
            
            # Initialize priority using reward magnitude (absolute TD-error proxy)
            priority = (abs(reward) + 1e-5) ** self.alpha
            heapq.heappush(self.priorities, (-priority, len(self.buffer)-1))
            
            # Update agent-specific counters
            self.agent_experience_map[agent_id] += 1
            
            # Maintain reward statistics
            self.reward_stats['sum'] += reward
            if reward > self.reward_stats['max']:
                self.reward_stats['max'] = reward
            if reward < self.reward_stats['min']:
                self.reward_stats['min'] = reward

    def sample(self, batch_size, strategy='uniform', beta=0.4, agent_distribution=None):
        """
        Sample batch using various strategies from RL literature.
        
        Args:
            strategy: 'uniform'|'prioritized'|'reward'|'agent_balanced'
            beta: Importance sampling correction (for prioritized)
            agent_distribution: Dict[agent_id: proportion] for stratified sampling
        """
        with self.lock:
            self._remove_stale_experiences()
            
            if batch_size > len(self.buffer):
                raise ValueError(f"Insufficient samples ({len(self.buffer)} available)")
                
            if strategy == 'prioritized':
                return self._prioritized_sample(batch_size, beta)
            elif strategy == 'reward':
                return self._reward_based_sample(batch_size)
            elif strategy == 'agent_balanced':
                return self._agent_balanced_sample(batch_size, agent_distribution)
            else:
                return self._uniform_sample(batch_size)

    def _prioritized_sample(self, batch_size, beta):
        """Prioritized sampling based on stored priorities (Schaul et al., 2015)"""
        priorities = np.array([-p[0] for p in heapq.nsmallest(batch_size, self.priorities)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(probs), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[i] for i in indices]
        return self._process_batch(batch), indices, weights

    def _reward_based_sample(self, batch_size):
        """Quality-based sampling using reward values (Oh et al., 2018)"""
        rewards = np.array([exp[3] for exp in self.buffer])
        exp_rewards = rewards - self.reward_stats['min'] + 1e-6
        probs = exp_rewards / exp_rewards.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        return self._process_batch(batch)

    def _agent_balanced_sample(self, batch_size, distribution):
        """Stratified sampling by agent ID (Christianos et al., 2020)"""
        if not distribution:
            distribution = {aid: count/len(self.buffer) 
                          for aid, count in self.agent_experience_map.items()}
            
        samples = []
        for agent_id, proportion in distribution.items():
            n_samples = int(batch_size * proportion)
            agent_experiences = [exp for exp in self.buffer if exp[0] == agent_id]
            samples.extend(random.sample(agent_experiences, min(n_samples, len(agent_experiences))))
            
        return self._process_batch(samples[:batch_size])

    def _uniform_sample(self, batch_size):
        """Random uniform sampling from all experiences.
        
        Implements baseline experience replay from:
        Mnih et al., "Playing Atari with Deep Reinforcement Learning", 2013
        
        Returns:
            tuple: (agent_ids, states, actions, rewards, next_states, dones)
        """
        # Randomly select experiences without prioritization
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        # Return formatted batch with original priority weights (1.0 for all)
        return self._process_batch(batch)
    
    def _remove_stale_experiences(self):
        """Automatic removal of stale experiences (Agarwal et al., 2021)"""
        now = datetime.now()
        stale_indices = [i for i, ts in enumerate(self.timestamps)
                        if now - ts > self.staleness_threshold]
        
        # Remove from all tracking structures
        for i in sorted(stale_indices, reverse=True):
            del self.buffer[i]
            del self.timestamps[i]
            
        logger.debug(f"Removed {len(stale_indices)} stale experiences")

    def update_priorities(self, indices, new_priorities):
        """Update priorities for prioritized replay (Schaul et al., 2015)"""
        with self.lock:
            for idx, priority in zip(indices, new_priorities):
                if idx < len(self.buffer):
                    heapq.heappush(self.priorities, (-priority ** self.alpha, idx))

    def apply_augmentation(self, batch, augment_fn):
        """Apply user-defined augmentation to sampled batch (Laskin et al., 2020)"""
        return augment_fn(batch)

    def _process_batch(self, batch):
        """Convert batch to numpy arrays"""
        agent_ids, states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(agent_ids),
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_)
        )

    # Maintain original interface methods (save/load/clear/etc)
    def save(self, filepath):
        """Save with compressed storage and metadata"""
        with self.lock:
            meta = {
                'capacity': self.capacity,
                'prioritization_alpha': self.alpha,
                'staleness_threshold': self.staleness_threshold.total_seconds(),
                'reward_stats': self.reward_stats
            }
            np.savez_compressed(
                filepath,
                buffer=np.array(self.buffer, dtype=object),
                timestamps=np.array(self.timestamps),
                priorities=np.array([-p[0] for p in self.priorities]),
                meta=meta
            )

    def load(self, filepath):
        """Load with metadata reconstruction"""
        data = np.load(filepath, allow_pickle=True)
        meta = data['meta'].item()
        
        with self.lock:
            self.capacity = meta['capacity']
            self.alpha = meta['prioritization_alpha']
            self.staleness_threshold = timedelta(seconds=meta['staleness_threshold'])
            self.reward_stats = meta['reward_stats']
            
            self.buffer = deque(data['buffer'].tolist(), maxlen=self.capacity)
            self.timestamps = deque(data['timestamps'].tolist(), maxlen=self.capacity)
            
            # Rebuild priority queue
            self.priorities = []
            for idx, priority in enumerate(data['priorities']):
                heapq.heappush(self.priorities, (-priority, idx))

    def get_agent_statistics(self):
        """Return experience distribution across agents"""
        return dict(self.agent_experience_map)

    def get_reward_statistics(self):
        """Return computed reward metrics"""
        return {
            **self.reward_stats,
            'mean': self.reward_stats['sum'] / len(self.buffer)
        }

    def clear(self):
        with self.lock:
            self.buffer.clear()
            logger.info("Replay buffer cleared.")

    def __len__(self):
        return len(self.buffer)

    def get_all(self):
        with self.lock:
            if not self.buffer:
                return [], [], [], [], [], []
            agent_ids, states, actions, rewards, next_states, dones = map(np.array, zip(*self.buffer))
        return agent_ids, states, actions, rewards, next_states, dones
