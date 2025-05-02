import os, sys
import yaml
import json
import heapq
import numpy as np
import random
from collections import deque, defaultdict
from threading import Lock
from datetime import datetime, timedelta
from src.utils.metrics_utils import FairnessMetrics, PerformanceMetrics, BiasDetection, MetricSummarizer
from logs.logger import get_logger

logger = get_logger("DistributedReplayBuffer")

class ExperienceReplayManager:
    def __init__(self, buffer, policy_net, batch_size=32):
        self.buffer = buffer
        self.policy_net = policy_net
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(
            batch_size,
            strategy='prioritized',
            beta=self.per_beta
        )
        # Unpack batch components
        states, actions, rewards, next_states, dones = batch
        
        # Calculate importance sampling weights
        weights = (len(self.replay_buffer) * self.replay_buffer.probs) ** -self.per_beta
        weights /= weights.max()

        # Train on batch
        losses = []
        for idx, (state, action, reward, next_state) in enumerate(zip(states, actions, rewards, next_states)):
            # Update network with importance weighting
            td_error = self._update_policy(state, action, reward, next_state)
            losses.append(td_error)
            
            # Update priorities in buffer
            new_priority = abs(td_error) + self.per_epsilon
            self.replay_buffer.update_priorities([idx], [new_priority])

        # Logging and memory consolidation
        self._consolidate_memory()
        logger.info(f"Prioritized replay complete. Avg loss: {np.mean(losses):.4f}")

    def sample_batch(self, batch_size):
        # Prioritized sampling logic
        pass

    def update_priorities(self, indices, errors):
        # Priority update logic
        pass


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

        # Fairness tracking
        self.fairness_stats = {
            'demographic_parity_violations': 0,
            'equalized_odds_violations': 0
        }
         
        # Academic provenance tracking
        self.metric_provenance = {
             'prioritized_sampling': "Schaul et al. (2015) Prioritized Experience Replay",
             'fairness_metrics': "Barocas et al. (2022) Fairness and Machine Learning"
         }
 
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
        
            # Track demographic distribution
            self._update_fairness_stats(agent_id, reward)

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

            # Get raw batch first
            if strategy == 'prioritized':
                raw_batch, indices, weights = self._prioritized_sample(batch_size, beta)
            elif strategy == 'reward':
                raw_batch = self._reward_based_sample(batch_size)
            elif strategy == 'agent_balanced':
                raw_batch = self._agent_balanced_sample(batch_size, agent_distribution)
            else:
                raw_batch = self._uniform_sample(batch_size)

            # Process batch for return
            processed_batch = self._process_batch(raw_batch)

            # Collect metrics from RAW batch before processing
            from src.utils.metrics_utils import PerformanceMetrics
            batch_metrics = {
                'calibration_error': PerformanceMetrics.calibration_error(
                    y_true=np.array([exp[3] for exp in raw_batch]),  # Use raw batch rewards
                    probs=np.array([abs(exp[3]) for exp in raw_batch])
                )
            }

            if hasattr(self, 'metric_bridge'):
                self.metric_bridge.submit_metrics(batch_metrics)

            # Return processed batch with numpy arrays
            return processed_batch

    def _update_fairness_stats(self, agent_id, reward):
        """Track metrics for autonomous bias detection"""
        with self.lock:
            # Track reward distribution per agent
            if agent_id not in self.reward_stats:
                self.reward_stats[agent_id] = []
            self.reward_stats[agent_id].append(reward)

    def _check_fairness(self, batch, strategy):
        """Implements Barocas's fairness framework for experience selection"""
        agent_ids = batch[0]
        unique, counts = np.unique(agent_ids, return_counts=True)
        selection_rates = {aid: count/len(agent_ids) for aid, count in zip(unique, counts)}
        
        # Demographic parity check
        violation, msg = FairnessMetrics.demographic_parity(
            sensitive_groups=list(selection_rates.keys()),
            positive_rates=selection_rates,
            threshold=0.1
        )
        
        if violation:
            self.fairness_stats['demographic_parity_violations'] += 1
            logger.warning(f"Fairness Alert: {msg}")
            
        # Reward distribution analysis
        reward_calibration = PerformanceMetrics.calibration_error(
            y_true=np.array([exp[3] for exp in self.buffer]),
            probs=np.array([abs(exp[3]) for exp in self.buffer])
        )
        logger.info(f"Reward calibration error: {reward_calibration:.4f}")

    def generate_health_report(self):
        """Autonomous self-assessment per Mitchell's model cards"""
        return MetricSummarizer.create_model_card(
            metrics={
                'fairness': self.fairness_stats,
                'performance': {
                    'reward_mean': np.mean([exp[3] for exp in self.buffer]),
                    'reward_variance': np.var([exp[3] for exp in self.buffer])
                }
            },
            references=self.metric_provenance
        )

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
                'reward_stats': self.reward_stats,
                'fairness_stats': self.fairness_stats,
                'metric_provenance': self.metric_provenance
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
            self.fairness_stats = meta.get('fairness_stats', {})
            self.metric_provenance = meta.get('metric_provenance', {})
            
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
