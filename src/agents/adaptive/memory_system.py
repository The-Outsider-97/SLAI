import numpy as np
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta

class MultiModalMemory:
    def __init__(self, config):
        from src.utils.replay_buffer import DistributedReplayBuffer
        
        # Episodic Memory (Time-based FIFO)
        self.episodic = deque(maxlen=config['episodic_capacity'])
        
        # Semantic Memory (Conceptual Knowledge with Decay)
        self.semantic = defaultdict(lambda: {
            'strength': 1.0,
            'last_accessed': datetime.now(),
            'data': None
        })
        
        # Forgetting Parameters
        self.staleness_threshold = timedelta(days=config['experience_staleness_days'])
        self.decay_rate = config['semantic_decay_rate']
        self.min_strength = config['min_memory_strength']
        
        # Replay Buffer for RL
        self.replay_buffer = DistributedReplayBuffer(
            config['replay_capacity'],
            config['priority_alpha'],
            self.staleness_threshold
        )

    def store_experience(self, state, action, reward):
        """Store experience with timestamp and initial strength"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now(),
            'strength': 1.0  # Initial memory strength
        }
        self.episodic.append(experience)
        self._update_semantic_memory(experience)

    def _update_semantic_memory(self, experience):
        """Convert significant experiences to semantic knowledge"""
        if abs(experience['reward']) > 1.0:
            key = f"state_{hash(experience['state']) % 1000}"
            self.semantic[key] = {
                'data': (experience['action'], experience['reward']),
                'strength': 1.0,
                'last_accessed': datetime.now()
            }

    def consolidate(self):
        """Apply forgetting mechanisms to all memory systems"""
        self._forget_old_episodes()
        self._decay_semantic_memory()
        self.replay_buffer._remove_stale_experiences()

    def _forget_old_episodes(self):
        """Time-based forgetting for episodic memory"""
        now = datetime.now()
        self.episodic = deque(
            [e for e in self.episodic 
             if (now - e['timestamp']) < self.staleness_threshold],
            maxlen=self.episodic.maxlen
        )

    def _decay_semantic_memory(self):
        """Decay and remove weak semantic memories"""
        for key in list(self.semantic.keys()):
            # Apply exponential decay
            self.semantic[key]['strength'] *= self.decay_rate
            # Remove if below threshold
            if self.semantic[key]['strength'] < self.min_strength:
                del self.semantic[key]

    def retrieve(self, query, recency_weight=0.7):
        """Unified retrieval with recency and strength weighting"""
        results = []
        
        # Episodic retrieval (time-weighted)
        for exp in reversed(self.episodic):
            if query in str(exp['state']):
                age = (datetime.now() - exp['timestamp']).days
                results.append({
                    'data': exp,
                    'score': recency_weight * (1/(age+1)) 
                            + (1-recency_weight) * exp['strength']
                })
        
        # Semantic retrieval (strength-weighted)
        if query in self.semantic:
            results.append({
                'data': self.semantic[query]['data'],
                'score': self.semantic[query]['strength']
            })
        
        # Return sorted by composite score
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def reinforce_memory(self, key, boost_factor=1.2):
        """Strengthen frequently accessed memories"""
        if key in self.semantic:
            self.semantic[key]['strength'] = min(
                self.semantic[key]['strength'] * boost_factor,
                1.0  # Maximum strength
            )
            self.semantic[key]['last_accessed'] = datetime.now()
