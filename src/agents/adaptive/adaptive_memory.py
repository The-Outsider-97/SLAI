
import json
import pandas as pd
import numpy as np
import hashlib
import signal
import sys

from typing import Dict, List, Optional, Union
from scipy.stats import entropy
from collections import defaultdict, deque
from datetime import datetime, timedelta

from src.utils.buffer.distributed_replay_buffer import DistributedReplayBuffer
from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.adaptive.utils.sgd_regressor import SGDRegressor
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Memory")
printer = PrettyPrinter

sys.stdout.flush()

class MultiModalMemory:
    """Reinforcement Learning Optimized Memory System with:
    - Policy parameter evolution tracking
    - Experience replay with self-tuning prioritization
    - Causal analysis of policy changes
    - Automated parameter tuning memory
    """
    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section('adaptive_memory')
        self.replay_capacity = self.memory_config.get('replay_capacity')
        self.drift_threshold = self.memory_config.get('drift_threshold')
        self.priority_alpha = self.memory_config.get('priority_alpha')
        self.retrieval_limit = self.memory_config.get('retrieval_limit')
        self.enable_goals = self.memory_config.get('enable_goals')
        self.goal_dim = self.memory_config.get('goal_dim')
        self.max_size = self.memory_config.get('max_size')
        self.goal_capacity = self.memory_config.get('goal_capacity')
        self.enable_policy_grad = self.memory_config.get('enable_policy_grad')
        self.uncertainty_dropout = self.memory_config.get('uncertainty_dropout')
        self.semantic_threshold = self.memory_config.get('semantic_threshold')
        
        self.sgd_config = get_config_section('sgd_regressor')
        self.learning_rate = self.sgd_config.get('learning_rate')

        self.rl_config = get_config_section('rl')
        self.action_dim = self.rl_config.get('action_dim')

        self.param_names = ['learning_rate', 'exploration_rate']

        # Core memory stores
        self.episodic = deque(maxlen=self.memory_config.get('episodic_capacity'))
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
        self.causal_model = SGDRegressor()

        # Forgetting Parameters
        self.staleness_threshold = timedelta(days=self.memory_config.get('experience_staleness_days'))
        self.min_strength = self.memory_config.get('min_memory_strength')
        self.decay_rate = self.memory_config.get('semantic_decay_rate',)

        # Replay Buffer
        self._init_drb()
        
        self.concept_drift_scores = []
        self.reward_sum = 0.0
        self.reward_count = 0
        self.max_abs_reward = 0.0

        signal.signal(signal.SIGINT, self._handle_emergency_exit)
        logger.info(f"Multi Modal Memory succesfully initialized...")

    def _init_drb(self):
        self.replay_buffer = DistributedReplayBuffer()

    def store_experience(self, state, action, reward, next_state=None, done=False, 
                         context: Optional[Dict] = None, params: Optional[Dict] = None,
                         **kwargs):
        """Store experience with timestamp and initial strength"""
        try:
            printer.status("INIT", "Experience storage succesfully initialized", "info")

            if reward is None or not isinstance(reward, (int, float)):
                logger.warning(f"Skipping experience due to invalid reward: {reward}")
                return None

            context_hash = self._hash_context(context) if context else self._fallback_context_hash(state, action)
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'timestamp': datetime.now(),
                'strength': 1.0,
                'context_hash': context_hash,
                # Store relevant parameters
                'params': {
                    'learning_rate': params.get('learning_rate') if params else None,
                    'exploration_rate': params.get('exploration_rate') if params else None
                } if params else {},
                **kwargs
            }

            self.episodic.append(experience)
            self._update_semantic_memory(experience)

            try:
                priority = self._calculate_priority(reward)
                self.replay_buffer.push(
                    agent_id="default", 
                    state=state, 
                    action=action, 
                    reward=reward, 
                    next_state=next_state, 
                    done=done,
                    priority=priority
                )
            except Exception as buffer_err:
                logger.warning(f"Replay buffer push failed: {buffer_err}")

            return experience

        except Exception as e:
            logger.error(f"Exception in store_experience: {e}")
            return None
        
    def clear_episodic(self):
        """Clear the episodic memory buffer"""
        self.episodic.clear()
        
    def _fallback_context_hash(self, state, action):
        fallback_str = f"state:{state}|action:{action}"
        return hashlib.sha256(fallback_str.encode()).hexdigest()

    def _hash_context(self, context: Dict) -> str:
        """Create a deterministic and collision-resistant hash from context."""
        try:
            if not context or not isinstance(context, dict):
                logger.warning("Context is missing or not a dictionary. Returning empty context_hash.")
                return ""
    
            # Sort keys for deterministic serialization
            context_str = str(sorted(context.items()))
            context_bytes = context_str.encode('utf-8')
    
            context_hash = hashlib.sha256(context_bytes).hexdigest()
            printer.status("INIT", f"Context hash generated: {context_hash[:12]}...", "info")
            return context_hash
        except Exception as e:
            logger.error(f"Failed to hash context: {e}", exc_info=True)
            return ""

    def _update_semantic_memory(self, experience: Dict):
        """Convert high-impact experiences to semantic knowledge"""
        printer.status("INIT", "Semantic memory update started", "info")

        try:
            reward = experience.get('reward')
            if reward is None or not isinstance(reward, (int, float)):
                logger.warning(f"Invalid reward encountered in experience: {reward}")
                return  # Skip invalid reward

            threshold = self.semantic_threshold or 0.5  # default fallback
            if abs(reward) > threshold:
                context_hash = experience.get('context_hash', '')
                if not context_hash:
                    printer.status("INVALID", f"Missing context_hash in experience with reward={reward}. Skipping semantic storage.", "warning")
                    return

                context_key = f"ctx_{context_hash[:6]}"
                self.semantic[context_key] = {
                    'data': (experience.get('action'), reward),
                    'strength': 1.0,
                    'last_accessed': datetime.now(),
                    'context_hash': context_hash
                }
                logger.info(f"Semantic memory updated with key: {context_key}")
        except Exception as e:
            logger.error(f"Exception in _update_semantic_memory: {e}")

    def _calculate_priority(self, reward: float) -> float:
        """
        Self-tuning priority calculation using reward normalization and adaptive scaling.
        Ensures fast failure and minimal blocking.
    
        Core Logic:
        1. Tracks cumulative absolute rewards.
        2. Calculates normalized reward.
        3. Measures deviation from running average.
        4. Applies non-linear scaling via tanh.
        5. Clamps output to [0, 1] for stability.
        """
        try:
            printer.status("INIT", "Priority calculation started", "info")
    
            # Validate reward
            if reward is None or not isinstance(reward, (int, float)):
                logger.warning(f"[PriorityCalc] Invalid reward type: {type(reward)}, value: {reward}")
                return 0.0
    
            # Use local copies to prevent threading/data race issues
            abs_reward = abs(reward)
            self.reward_sum += abs_reward
            self.reward_count += 1
            self.max_abs_reward = max(self.max_abs_reward, abs_reward)
    
            # Safe denominators
            norm_denom = self.max_abs_reward if self.max_abs_reward > 1e-6 else 1e-6
            count_denom = self.reward_count if self.reward_count > 0 else 1
    
            # Core calculations
            normalized_reward = abs_reward / norm_denom
            avg_reward = self.reward_sum / count_denom
            reward_deviation = abs_reward - avg_reward
            deviation_factor = 1.0 + np.tanh(reward_deviation / (avg_reward + 1e-6))
    
            raw_priority = (normalized_reward * deviation_factor + 0.01) ** self.priority_alpha
            clipped_priority = float(min(max(raw_priority, 0.0), 1.0))
    
            printer.status("INIT", f"Calculated priority: {clipped_priority:.4f}", "success")
            return clipped_priority
    
        except Exception as e:
            logger.error(f"Exception in _calculate_priority: {e}", exc_info=True)
            # In failure, still return low but valid priority
            return 0.01

    def log_parameters(self, performance: float, params: Dict):
        """Track evolution of learning parameters"""
        printer.status("INIT", "Param logger succesfully initialized", "info")

        entry = {
            'timestamp': datetime.now(),
            'learning_rate': float(params.get('learning_rate', np.nan)),
            'exploration_rate': float(params.get('exploration_rate', np.nan)),
            'discount_factor': float(params.get('discount_factor', np.nan)),
            'temperature': float(params.get('temperature', np.nan)),
            'performance': float(performance)
        }
        self.parameter_evolution = pd.concat([
            self.parameter_evolution, 
            pd.DataFrame([entry])
        ], ignore_index=True)

    def analyze_parameter_impact(self, window_size: int = 100) -> Dict:
        """Analyze relationships between parameter changes and performance"""
        printer.status("INIT", "Param impact succesfully initialized", "info")

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
    

    def get_memory_report(self) -> Dict:
        """Generate unified memory analysis report"""
        printer.status("INIT", "Reporter mem succesfully initialized", "info")

        return {
            'parameter_analysis': self._analyze_parameters(),
            'intervention_impact': self._intervention_statistics(),
            'drift_status': self.detect_drift(),
            'replay_stats': self.replay_buffer.stats(),
            'semantic_summary': self._semantic_analysis(),
            'avg_strength': (
                np.mean([v['strength'] for v in self.semantic.values()])
                if self.semantic else 0.0
            ),
        }

    def _analyze_parameters(self) -> Dict:
        """Statistical analysis of parameter evolution"""
        return self.parameter_evolution.describe().to_dict()

    def _intervention_statistics(self) -> Dict:
        """Summarize policy intervention effectiveness"""
        if not self.policy_interventions:
            return {}

    def detect_drift(self, window_size: int = 30) -> bool:
        """Performance-based concept drift detection"""
        if len(self.parameter_evolution) < 2*window_size:
            return kl_div > self.drift_threshold
            
        recent = self.parameter_evolution['performance'][-window_size:].values
        historical = self.parameter_evolution['performance'][-2*window_size:-window_size].values
        
        p = np.histogram(recent, bins=10)[0] + 1e-6
        q = np.histogram(historical, bins=10)[0] + 1e-6
        kl_div = entropy(p, q)
        
        self.concept_drift_scores.append(kl_div)
        return kl_div > self.drift_threshold

    def _semantic_analysis(self) -> Dict:
        """Analyze semantic memory characteristics"""
        return {
            'total_concepts': len(self.semantic),
            'avg_strength': np.mean([v['strength'] for v in self.semantic.values()]),
            'active_contexts': len(set(v['context_hash'] for v in self.semantic.values()))
        }

    def apply_policy_intervention(self, intervention: Dict, effect: Dict):
        """Log policy changes and their effects"""
        printer.status("INIT", "Policy intervention succesfully initialized", "info")

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

    def _update_causal_model(self, intervention: Dict, effect: Dict):
        """Update causal relationships between policy changes and outcomes"""
        before = intervention['params_before']
        after = intervention['params_after']
        
        X = np.array([
            before.get('learning_rate', 0.0),
            before.get('exploration_rate', 0.0),
            after.get('learning_rate', before.get('learning_rate', 0.0)) - before.get('learning_rate', 0.0),
            after.get('exploration_rate', before.get('exploration_rate', 0.0)) - before.get('exploration_rate', 0.0)
        ]).reshape(1, -1)
        
        y = np.array([effect['performance_delta']])
        self.causal_model.partial_fit(X, y)
            
        df = pd.DataFrame(self.policy_interventions)
        return df.groupby('type').agg({
            'effect_size': ['mean', 'std'],
            'causal_impact': 'median'
        }).to_dict()

    def consolidate(self):
        """Apply forgetting mechanisms to all memory systems with parameter-aware forgetting"""
        self._forget_old_episodes()
        self._decay_semantic_memory()
        self.replay_buffer._remove_stale_experiences()
        self._prune_parameter_history()

    def _forget_old_episodes(self):
        """Remove stale episodes based on time and memory strength"""
        if not self.episodic:
            return
            
        now = datetime.now()
        new_episodic = deque(maxlen=self.episodic.maxlen)
        removed_count = 0
        
        for exp in self.episodic:
            # Calculate age-based decay factor (linear decay over staleness threshold)
            age = now - exp['timestamp']
            age_factor = max(0, 1 - age.total_seconds() / self.staleness_threshold.total_seconds())
            
            # Apply combined decay (age + configured decay rate)
            exp['strength'] *= self.decay_rate * age_factor
            
            # Keep only sufficiently strong and recent memories
            if (exp['strength'] > self.min_strength and 
                age <= self.staleness_threshold):
                new_episodic.append(exp)
            else:
                removed_count += 1
        
        self.episodic = new_episodic
        logger.debug(f"Forgot {removed_count} stale episodes")

    def _decay_semantic_memory(self):
        """Decay and remove weak semantic memories"""
        for key in list(self.semantic.keys()):
            # Apply exponential decay
            self.semantic[key]['strength'] *= self.decay_rate
            # Remove if below threshold
            if self.semantic[key]['strength'] < self.min_strength:
                del self.semantic[key]

    def _prune_parameter_history(self):
        """Remove outdated parameter records"""
        # Calculate cutoff using timedelta directly
        cutoff = datetime.now() - self.staleness_threshold
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

        limit = self.retrieval_limit
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

    def _extract_parameter_features(self, query: str) -> Dict[str, float]:
        """Convert text query to parameter space features using keyword analysis"""
        # Initialize with default config values
        config = get_config_section('parameter_tuner')
        features = {
            'learning_rate': config.get('base_learning_rate', 0.001),
            'exploration_rate': config.get('base_exploration_rate', 0.1),
            'discount_factor': config.get('base_discount_factor', 0.95),
            'temperature': config.get('base_temperature', 1.0)
        }
        
        # Use recent parameter history if available
        if not self.parameter_evolution.empty:
            recent = self.parameter_evolution.iloc[-1]
            # Only update if values are not None
            if not pd.isnull(recent['learning_rate']):
                features['learning_rate'] = recent['learning_rate']
            if not pd.isnull(recent['exploration_rate']):
                features['exploration_rate'] = recent['exploration_rate']
            if not pd.isnull(recent['discount_factor']):
                features['discount_factor'] = recent['discount_factor']
            if not pd.isnull(recent['temperature']):
                features['temperature'] = recent['temperature']
        
        # Process query keywords
        query = query.lower()
        modifiers = {
            'high': 1.5, 'increase': 1.3, 'boost': 1.2,
            'low': 0.5, 'reduce': 0.7, 'decrease': 0.8
        }
        
        # Apply modifiers to parameters mentioned in query
        for param in features:
            if param.replace('_', ' ') in query:
                for mod, factor in modifiers.items():
                    if mod in query:
                        features[param] = np.clip(
                            features[param] * factor,
                            self.config['parameter_tuner'].get(f'min_{param}', 0.0001),
                            self.config['parameter_tuner'].get(f'max_{param}', 1.0)
                        )
                        break
        
        return features

    def _calculate_parameter_similarity(self, experience: Dict, target: Dict) -> float:
        """Compute similarity between experience and target parameters"""
        # Initialize difference sum
        total_diff = 0.0
        
        # Calculate differences for each parameter
        for param in self.param_names:
            exp_val = experience['params'].get(param, 0.0) or 0.0
            target_val = target.get(param, 0.0)
            total_diff += abs(exp_val - target_val)
        
        # Return similarity score (avoid division by zero)
        return 1.0 / (1.0 + total_diff) if total_diff > 0 else 1.0
    
    def _generate_context_hash(self, context: dict) -> str:
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
    
        sanitized = sanitize(context)
        return hashlib.md5(json.dumps(sanitized, sort_keys=True).encode()).hexdigest()
    
    def _generate_memory_bias(self, memories: List[Dict]) -> np.ndarray:
        """
        Generate action bias vector from retrieved memories.
        Combines semantic and episodic memories into a unified bias signal.
        
        Args:
            memories: List of memory dictionaries from retrieve()
            
        Returns:
            Bias vector with length = action_dim
        """
        # Initialize bias vector
        bias = np.zeros(self.action_dim)
        
        if not memories:
            return bias
        
        # Process semantic memories (contextual knowledge)
        semantic_bias = np.zeros(self.action_dim)
        semantic_weight = 0.0
        
        # Process episodic memories (specific experiences)
        episodic_bias = np.zeros(self.action_dim)
        episodic_weight = 0.0
        
        for memory in memories:
            mem_type = memory['type']
            score = memory['score']
            data = memory['data']
            
            if mem_type == 'semantic':
                # Semantic memory: (action, reward) tuple
                action, reward = data
                if 0 <= action < self.action_dim:
                    # Weight by memory strength and recency
                    semantic_bias[action] += reward * score
                    semantic_weight += abs(reward) * score
                    
            elif mem_type == 'episodic':
                # Episodic memory: full experience dictionary
                action = data['action']
                reward = data['reward']
                if 0 <= action < self.action_dim:
                    # Scale by memory relevance score
                    episodic_bias[action] += reward * score
                    episodic_weight += abs(reward) * score
        
        # Normalize and combine memory components
        final_bias = np.zeros(self.action_dim)
        
        if semantic_weight > 0:
            final_bias += semantic_bias / semantic_weight
            
        if episodic_weight > 0:
            final_bias += episodic_bias / episodic_weight
        
        # Apply softmax to create relative preferences
        max_val = np.max(final_bias) if np.any(final_bias != 0) else 1.0
        scaled_bias = final_bias - max_val  # For numerical stability
        exp_bias = np.exp(scaled_bias)
        softmax_bias = exp_bias / np.sum(exp_bias)
        
        # Scale to moderate influence (range: -1 to 1)
        return 2.0 * (softmax_bias - 0.5)
    
    def sample(self, batch_size: int):
        """
        Sample a batch from the replay buffer and return it in a structured format.
        """
        batch = self.replay_buffer.sample(batch_size)
    
        # Extract first 5 fields (state, action, reward, next_state, done)
        trimmed_batch = [sample[:5] for sample in batch]
        states, actions, rewards, next_states, dones = zip(*trimmed_batch)
    
        # Convert states to proper numeric format
        clean_states = [np.array(s, dtype=np.float32) for s in states]
    
        # Dummy advantage computation
        advantages = [r for r in rewards]
    
        return {
            "states": clean_states,
            "actions": actions,
            "advantages": advantages
        }

    def size(self) -> Dict[str, int]:
        """Get comprehensive memory size report"""
        return {
            'episodic': len(self.episodic),
            'semantic': len(self.semantic),
            'parameter_history': len(self.parameter_evolution),
            'policy_interventions': len(self.policy_interventions),
            'replay_buffer': len(self.replay_buffer),
            'concept_drift_scores': len(self.concept_drift_scores),
            'total': (
                len(self.episodic) + 
                len(self.semantic) + 
                len(self.parameter_evolution) +
                len(self.policy_interventions)
            )
        }

    def reinforce_memory(self, key, boost_factor=1.2):
        """Strengthen frequently accessed memories"""
        if key in self.semantic:
            self.semantic[key]['strength'] = min(
                self.semantic[key]['strength'] * boost_factor,
                1.0  # Maximum strength
            )
            self.semantic[key]['last_accessed'] = datetime.now()

    def _handle_emergency_exit(self, signum, frame):
        logger.critical("EMERGENCY EXIT - Forcing buffer unlock")
        self.replay_buffer.lock.release()
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime, timedelta
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
    
    printer.pretty("Memory report:", memory.get_memory_report(), "success")

    print("\n* * * * * Phase 2 * * * * *\n")
    state=2
    action=4
    reward=1.5
    context=None

    store = memory.store_experience(state=state, action=action, reward=reward, context=context)

    # printer.pretty("EXP", store, "success")
    printer.pretty("EXP", store, "success")
    print("\n=== Successfully Ran Multi Modal Memory ===\n")
