
import pandas as pd
import numpy as np
import hashlib

from typing import Dict, List, Optional, Union
from scipy.stats import entropy
from collections import defaultdict, deque
from datetime import datetime, timedelta

from src.utils.buffer.distributed_replay_buffer import DistributedReplayBuffer
from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.adaptive.utils.sgd_regressor import SGDRegressor
from logs.logger import get_logger

logger = get_logger("Adaptive Memory")

class MultiModalMemory:
    """Reinforcement Learning Optimized Memory System with:
    - Policy parameter evolution tracking
    - Experience replay with self-tuning prioritization
    - Causal analysis of policy changes
    - Automated parameter tuning memory
    """
    def __init__(self):
        self.config = load_global_config()
        self.rl_config = get_config_section('adaptive_memory')

        # Core memory stores
        self.episodic = deque(maxlen=self.rl_config.get('episodic_capacity', 1000))
        self.parameter_evolution = pd.DataFrame(columns=[  # Initialize DataFrame
            'timestamp', 'learning_rate', 'exploration_rate', 
            'discount_factor', 'temperature', 'performance'
        ])
        self.policy_interventions = []

        # Semantic Memory
        self.semantic = defaultdict(lambda: {
            'strength': 1.0,
            'last_accessed': datetime.now(),
            'data': None,
            'context_hash': ''
        })
        self.causal_model = SGDRegressor(eta0=0.01, learning_rate='constant')

        # Replay Buffer
        self._init_drb()

        # Forgetting Parameters
        self.staleness_threshold = timedelta(days=self.rl_config.get('experience_staleness_days', 7))
        self.decay_rate = self.rl_config.get('semantic_decay_rate', 0.9)
        self.min_strength = self.rl_config.get('min_memory_strength', 0.1)
        
        self.concept_drift_scores = []

    def _init_drb(self):
        self.replay_buffer = DistributedReplayBuffer()

    def store_experience(self, state, action, reward, context: Optional[Dict] = None):
        """Store experience with timestamp and initial strength"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now(),
            'strength': 1.0,  # Initial memory strength
            'context_hash': self._hash_context(context) if context else ''
        }
        self.episodic.append(experience)
        self._update_semantic_memory(experience)
        self.replay_buffer.add(experience, priority=self._calculate_priority(reward))

    def log_parameters(self, performance: float, params: Dict):
        """Track evolution of learning parameters"""
        entry = {
            'timestamp': datetime.now(),
            'learning_rate': params.get('learning_rate'),
            'exploration_rate': params.get('exploration_rate'),
            'discount_factor': params.get('discount_factor'),
            'temperature': params.get('temperature'),
            'performance': performance
        }
        self.parameter_evolution = pd.concat([
            self.parameter_evolution, 
            pd.DataFrame([entry])
        ], ignore_index=True)

    def apply_policy_intervention(self, intervention: Dict, effect: Dict):
        """Log policy changes and their effects"""
        intervention_record = {
            'timestamp': datetime.now(),
            'type': intervention.get('type'),
            'params_before': intervention.get('params_before'),
            'params_after': intervention.get('params_after'),
            'effect_size': effect.get('performance_delta'),
            'causal_impact': None
        }
        self.policy_interventions.append(intervention_record)
        self._update_causal_model(intervention_record, effect)

    def analyze_parameter_impact(self, window_size: int = 100) -> Dict:
        """Analyze relationships between parameter changes and performance"""
        if len(self.parameter_evolution) < window_size:
            return {}
            
        recent = self.parameter_evolution.iloc[-window_size:]
        X = recent[['learning_rate', 'exploration_rate', 'discount_factor', 'temperature']].values
        y = recent['performance'].values
        
        self.causal_model.partial_fit(X, y)
        return {
            'learning_rate_impact': self.causal_model.coef_[0],
            'exploration_impact': self.causal_model.coef_[1],
            'discount_impact': self.causal_model.coef_[2],
            'temperature_impact': self.causal_model.coef_[3]
        }

    def _calculate_priority(self, reward: float) -> float:
        """Self-tuning priority calculation"""
        alpha = self.rl_config.get('priority_alpha', 0.6)
        return (abs(reward) + 0.01) ** alpha
    

    def detect_drift(self, window_size: int = 30) -> bool:
        """Performance-based concept drift detection"""
        if len(self.parameter_evolution) < 2*window_size:
            return kl_div > self.rl_config.get('drift_threshold', 0.4)
            
        recent = self.parameter_evolution['performance'][-window_size:].values
        historical = self.parameter_evolution['performance'][-2*window_size:-window_size].values
        
        p = np.histogram(recent, bins=10)[0] + 1e-6
        q = np.histogram(historical, bins=10)[0] + 1e-6
        kl_div = entropy(p, q)
        
        self.concept_drift_scores.append(kl_div)
        return kl_div > self.config.get('drift_threshold', 0.4)

    def get_memory_report(self) -> Dict:
        """Generate unified memory analysis report"""
        return {
            'parameter_analysis': self._analyze_parameters(),
            'intervention_impact': self._intervention_statistics(),
            'drift_status': self.detect_drift(),
            'replay_stats': self.replay_buffer.stats(),
            'semantic_summary': self._semantic_analysis()
        }

    def _update_semantic_memory(self, experience: Dict):
        """Convert high-impact experiences to semantic knowledge"""
        threshold = self.rl_config.get('semantic_threshold', 0.8)
        if abs(experience['reward']) > threshold:
            context_key = f"ctx_{experience['context_hash'][:6]}"
            self.semantic[context_key] = {
                'data': (experience['action'], experience['reward']),
                'strength': 1.0,
                'last_accessed': datetime.now(),
                'context_hash': experience['context_hash']
            }

    def _hash_context(self, context: Dict) -> str:
        """Create unique context identifier"""
        return hashlib.sha256(str(context).encode()).hexdigest()

    def _update_causal_model(self, intervention: Dict, effect: Dict):
        """Update causal relationships between policy changes and outcomes"""
        X = np.array([
            intervention['params_before']['learning_rate'],
            intervention['params_before']['exploration_rate'],
            intervention['params_after']['learning_rate'] - intervention['params_before']['learning_rate'],
            intervention['params_after']['exploration_rate'] - intervention['params_before']['exploration_rate']
        ]).reshape(1, -1)
        
        y = np.array([effect['performance_delta']])
        self.causal_model.partial_fit(X, y)

    def _analyze_parameters(self) -> Dict:
        """Statistical analysis of parameter evolution"""
        return self.parameter_evolution.describe().to_dict()

    def _intervention_statistics(self) -> Dict:
        """Summarize policy intervention effectiveness"""
        if not self.policy_interventions:
            return {}
            
        df = pd.DataFrame(self.policy_interventions)
        return df.groupby('type').agg({
            'effect_size': ['mean', 'std'],
            'causal_impact': 'median'
        }).to_dict()

    def _semantic_analysis(self) -> Dict:
        """Analyze semantic memory characteristics"""
        return {
            'total_concepts': len(self.semantic),
            'avg_strength': np.mean([v['strength'] for v in self.semantic.values()]),
            'active_contexts': len(set(v['context_hash'] for v in self.semantic.values()))
        }

    def consolidate(self):
        """Apply forgetting mechanisms to all memory systems with parameter-aware forgetting"""
        self._forget_old_episodes()
        self._decay_semantic_memory()
        self.replay_buffer._remove_stale_experiences()
        self._prune_parameter_history()

    def _forget_old_episodes(self):
        pass

    def _prune_parameter_history(self):
        """Remove outdated parameter records"""
        retention_days = self.rl_config.get('param_retention_days', 7)
        cutoff = datetime.now() - timedelta(days=retention_days)
        self.parameter_evolution = self.parameter_evolution[
            self.parameter_evolution['timestamp'] > cutoff
        ]

    def retrieve(self, query, context: Optional[Dict] = None,):
        """Context-aware retrieval with parameter prioritization"""
        results = []
        
        # Contextual semantic retrieval
        if context:
            context_hash = self._hash_context(context)
            semantic_key = f"ctx_{context_hash[:6]}"
            if semantic_key in self.semantic:
                results.append({
                    'data': self.semantic[semantic_key]['data'],
                    'score': self.semantic[semantic_key]['strength'],
                    'type': 'semantic'
                })

        # Parameter-relevant episodic retrieval
        param_features = self._extract_parameter_features(query)
        for exp in reversed(self.episodic):
            similarity = self._calculate_parameter_similarity(exp, param_features)
            results.append({
                'data': exp,
                'score': similarity,
                'type': 'episodic'
            })

        limit = self.rl_config.get('retrieval_limit', 5)
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

    def _extract_parameter_features(self, query: str) -> Dict:
        """Convert text query to parameter space features"""
        # Implementation depends on specific query processing
        return {
            'learning_rate': 0.001,
            'exploration_rate': 0.001
        }

    def size(self):
        return len(self.memory)

    def _calculate_parameter_similarity(self, experience: Dict, target: Dict) -> float:
        """Compute similarity between experience and target parameters"""
        lr_diff = abs(experience.get('learning_rate', 0) - target['learning_rate'])
        exp_diff = abs(experience.get('exploration_rate', 0) - target['exploration_rate'])
        return 1.0 / (1.0 + lr_diff + exp_diff)

    def _decay_semantic_memory(self):
        """Decay and remove weak semantic memories"""
        for key in list(self.semantic.keys()):
            # Apply exponential decay
            self.semantic[key]['strength'] *= self.decay_rate
            # Remove if below threshold
            if self.semantic[key]['strength'] < self.min_strength:
                del self.semantic[key]

    def reinforce_memory(self, key, boost_factor=1.2):
        """Strengthen frequently accessed memories"""
        if key in self.semantic:
            self.semantic[key]['strength'] = min(
                self.semantic[key]['strength'] * boost_factor,
                1.0  # Maximum strength
            )
            self.semantic[key]['last_accessed'] = datetime.now()

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import time
    
    test_config = {
        'adaptive_memory': {
            'episodic_capacity': 1000,
            'semantic_threshold': 0.8,
            'priority_alpha': 0.6,
            'param_retention_days': 7,
            'drift_threshold': 0.4,
            'retrieval_limit': 5,
            'replay_capacity': 100000
    }}
    
    memory = MultiModalMemory()
    
    # Simulate RL parameter evolution
    for i in range(100):
        params = {
            'learning_rate': 0.01 * (1 - i/100),
            'exploration_rate': 0.3 * (0.95 ** i),
            'discount_factor': 0.95,
            'temperature': 1.0 - (i*0.005)
        }
        memory.log_parameters(performance=np.random.normal(0.8, 0.1), params=params)
    
    print("Memory report:", memory.get_memory_report())
