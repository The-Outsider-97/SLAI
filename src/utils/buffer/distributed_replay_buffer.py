
import time
import heapq
import yaml, json
import numpy as np
import random

from threading import Lock
from datetime import timedelta, datetime
from collections import deque, defaultdict

from src.utils.metrics_utils import FairnessMetrics, PerformanceMetrics, BiasDetection, MetricSummarizer
from logs.logger import get_logger

logger = get_logger("Distributed Replay Buffer")

CONFIG_PATH = "src/utils/buffer/configs/buffer_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config: base_config.update(user_config)
    return base_config

class DistributedReplayBuffer:
    def __init__(self, user_config=None):
        """
        Enhanced replay buffer with features from RL research literature.
        
        Args:
            capacity: Maximum number of transitions stored
            staleness_threshold: Automatically remove experiences older than this
            prioritization_alpha: Prioritization exponent (0=uniform, 1=full prioritization)
        """
        config = get_merged_config(user_config)
        dist_config = config.get('distributed', {})
        
        self.capacity = dist_config.get('capacity', 100_000)
        self.staleness_threshold = timedelta(days=dist_config.get('staleness_threshold_days', 1))
        self.alpha = dist_config.get('prioritization_alpha', 0.6)
        seed = dist_config.get('seed', None)

        # Core storage
        self.buffer = deque(maxlen=self.capacity)
        self.lock = Lock()
        self.priorities = []
        self.timestamps = deque(maxlen=self.capacity)
        self.agent_experience_map = defaultdict(int)
        self.reward_stats = {'sum': 0.0, 'max': -np.inf, 'min': np.inf}
        
        # Seed handling
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        logger.info(f"Initialized distributed buffer (Capacity: {self.capacity}, alpha: {self.alpha})")

    def push(self, agent_id, state, action, reward, next_state, done):
        """
        Store transition with automatic priority initialization and quality tracking.
        Implements prioritized experience replay (Schaul et al., 2015).
        """
        with self.lock:
            experience = (agent_id, state, action, reward, next_state, done)
            self.buffer.append(experience)
            self.timestamps.append(datetime.now())
            
            # Calculate initial priority using reward magnitude
            priority = (abs(reward) + 1e-5) ** self.alpha
            heapq.heappush(self.priorities, (-priority, len(self.buffer) - 1))
            
            # Update agent experience count
            self.agent_experience_map[agent_id] += 1
            
            # Update global reward statistics
            self.reward_stats['sum'] += reward
            if reward > self.reward_stats['max']:
                self.reward_stats['max'] = reward
            if reward < self.reward_stats['min']:
                self.reward_stats['min'] = reward
            
            # Track per-agent rewards in a separate structure
            if not hasattr(self, 'agent_rewards'):
                self.agent_rewards = defaultdict(list)
            self.agent_rewards[agent_id].append(reward)
            
            # Update fairness metrics
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
            
            if len(self.buffer) < batch_size:
                raise ValueError(f"Insufficient samples ({len(self.buffer)} available)")
            
            if strategy == 'prioritized':
                processed_batch, indices, weights = self._prioritized_sample(batch_size, beta)
            elif strategy == 'reward':
                processed_batch = self._reward_based_sample(batch_size)
            elif strategy == 'agent_balanced':
                processed_batch = self._agent_balanced_sample(batch_size, agent_distribution)
            else:
                processed_batch = self._uniform_sample(batch_size)
            
            # Compute calibration error
            try:
                rewards = processed_batch[3]
                probs = np.abs(rewards)
                calibration = PerformanceMetrics.calibration_error(y_true=rewards, probs=probs)
                logger.info(f"Calibration error: {calibration}")
            except Exception as e:
                logger.warning(f"Failed to compute calibration error: {e}")
            
            return processed_batch

    def _prioritized_sample(self, batch_size, beta):
        # Filter valid priorities (indices within current buffer)
        valid_priorities = []
        valid_indices = []
        for p in self.priorities:
            priority, idx = -p[0], p[1]
            if idx < len(self.buffer):
                valid_priorities.append(priority)
                valid_indices.append(idx)
        
        if not valid_priorities:
            raise ValueError("No valid priorities available for sampling")
        
        priorities = np.array(valid_priorities)
        sum_priorities = priorities.sum()
        
        if sum_priorities <= 0:
            probs = np.ones(len(priorities)) / len(priorities)
        else:
            probs = priorities / sum_priorities
        
        if len(valid_priorities) < batch_size:
            raise ValueError(f"Not enough valid samples for prioritized sampling")
        
        selected_indices = np.random.choice(valid_indices, size=batch_size, p=probs)
        batch = [self.buffer[i] for i in selected_indices]
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[np.isin(valid_indices, selected_indices)]) ** (-beta)
        if weights.size > 0:
            weights /= weights.max()
        else:
            weights = np.ones(batch_size)
        
        return self._process_batch(batch), selected_indices, weights
    
    def _reward_based_sample(self, batch_size):
        rewards = np.array([exp[3] for exp in self.buffer])
        
        if self.reward_stats['min'] == np.inf or self.reward_stats['max'] == -np.inf:
            # Uniform sampling if no valid rewards
            return self._uniform_sample(batch_size)
        
        min_reward = self.reward_stats['min']
        exp_rewards = rewards - min_reward + 1e-6
        sum_exp = exp_rewards.sum()
        
        if sum_exp <= 0:
            probs = np.ones(len(rewards)) / len(rewards)
        else:
            probs = exp_rewards / sum_exp
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        return self._process_batch(batch)
    
    def _agent_balanced_sample(self, batch_size, distribution):
        if not distribution:
            total = sum(self.agent_experience_map.values())
            if total == 0:
                return self._uniform_sample(batch_size)
            distribution = {aid: count / total for aid, count in self.agent_experience_map.items()}
        
        samples = []
        for agent_id, prop in distribution.items():
            n_samples = int(batch_size * prop)
            agent_exps = [exp for exp in self.buffer if exp[0] == agent_id]
            if not agent_exps:
                continue
            n = min(n_samples, len(agent_exps))
            samples.extend(random.sample(agent_exps, n))
        
        # Fill remaining samples if needed
        if len(samples) < batch_size:
            remaining = batch_size - len(samples)
            samples.extend(random.sample(self.buffer, remaining))
        elif len(samples) > batch_size:
            samples = samples[:batch_size]
        
        return self._process_batch(samples)
    

    def _remove_stale_experiences(self):
        """Efficient stale experience removal for deque."""
        now = datetime.now()
        new_buffer = deque()
        new_timestamps = deque()

        for exp, ts in zip(self.buffer, self.timestamps):
            if now - ts <= self.staleness_threshold:
                new_buffer.append(exp)
                new_timestamps.append(ts)

        removed = len(self.buffer) - len(new_buffer)
        self.buffer = new_buffer
        self.timestamps = new_timestamps

        logger.debug(f"Removed {removed} stale experiences")

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
            #del self.buffer[i]
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


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Distributed Replay Buffer ===\n")
    user_config = None
    config = get_merged_config(user_config)

    buffer = DistributedReplayBuffer(user_config)

    print(f"\n{buffer}")

    print("\n* * * * * Phase 2 * * * * *\n")

    print("\n* * * * * Phase 3 * * * * *\n")
#    agent_id1='dqn'
#    agent_id2='maml'
#    cross = factory._crossover(agent_id1, agent_id2)
#    print(f"\n{cross}")

    print("\n* * * * * Phase 4 * * * * *\n")
#    monitor = factory.monitor_architecture()
#    print(f"\n{monitor}")

    print("\n=== Successfully Ran Distributed Replay Buffer ===\n")
