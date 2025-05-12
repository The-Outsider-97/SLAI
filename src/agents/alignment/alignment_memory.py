"""
Continual Alignment Memory System
Implements:
- Causal outcome tracing (Goyal et al., 2019)
- Experience replay for alignment (Parisi et al., 2019)
- Concept drift detection
- Intervention effect tracking
"""
import yaml
import hashlib
import numpy as np
import pandas as pd

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy.stats import pearsonr, entropy
# rom sklearn.covariance import EmpiricalCovariance # if self.causal_model = EmpiricalCovariance()
from sklearn.linear_model import SGDRegressor

from logs.logger import get_logger

logger = get_logger("Alignment Memory")

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: str, config_file_path: str):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class AlignmentMemory:
    """
    Persistent alignment memory module with:
    - Temporal outcome logging
    - Causal intervention analysis
    - Experience replay for training
    - Concept drift detection
    
    Memory Structure:
    1. Alignment Logs: Raw evaluation metrics over time
    2. Context Registry: Domain-specific outcome statistics
    3. Intervention Graph: Corrections and their effects
    4. Causal Model: Learned relationships between actions/outcomes
    """

    def __init__(self,
                 config_section_name: str = "alignment_memory",
                 config_file_path: str = "src/agents/alignment/configs/alignment_config.yaml"
                 ):
        self.config = get_config_section(config_section_name, config_file_path)
        
        # Core memory stores
        self.alignment_logs = pd.DataFrame(columns=[
            'timestamp', 'metric', 'value', 'threshold', 'violation', 'context'
        ])
        self.context_registry = pd.DataFrame(columns=[
            'context_hash', 'bias_rate', 'ethics_violations', 'last_updated'
        ])
        self.intervention_graph = []
        # self.causal_model = EmpiricalCovariance()
        self.causal_model = SGDRegressor(eta0=0.01, learning_rate='constant')
        
        # State tracking
        self.concept_drift_scores = []
        self.replay_buffer = []

    def log_evaluation(self, metric: str, value: float, 
                      threshold: float, context: Dict) -> None:
        """Record alignment evaluation outcome"""
        entry = {
            'timestamp': datetime.now(),
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'violation': value > threshold,
            'context': self._hash_context(context)
        }
        self.alignment_logs = pd.concat(
            [self.alignment_logs, pd.DataFrame([entry])], 
            ignore_index=True
        )
        self._update_replay_buffer(entry)
        self._update_context_registry(context, value > threshold)

    def record_outcome(self, context: Dict, outcome: Dict) -> None:
        """Store context-specific operational outcomes"""
        context_hash = self._hash_context(context)
        now = datetime.now()
        
        # Update or create context entry
        if context_hash in self.context_registry['context_hash'].values:
            idx = self.context_registry[self.context_registry.context_hash == context_hash].index
            self.context_registry.loc[idx, 'last_updated'] = now
            self.context_registry.loc[idx, 'bias_rate'] = outcome.get('bias_rate', np.nan)
            self.context_registry.loc[idx, 'ethics_violations'] = outcome.get('ethics_violations', 0)
        else:
            new_entry = {
                'context_hash': context_hash,
                'bias_rate': outcome.get('bias_rate', 0),
                'ethics_violations': outcome.get('ethics_violations', 0),
                'last_updated': now
            }
            self.context_registry = pd.concat(
                [self.context_registry, pd.DataFrame([new_entry])],
                ignore_index=True
            )

    def apply_correction(self, correction: Dict, effect: Dict) -> None:
        """Log intervention and its observed effects"""
        intervention = {
            'timestamp': datetime.now(),
            'type': correction.get('type'),
            'magnitude': correction.get('magnitude'),
            'target': correction.get('target'),
            'pre_state': self._get_current_state(),
            'post_state': effect,
            'causal_impact': None  # Populated by analyze_causes()
        }
        self.intervention_graph.append(intervention)
        self._update_causal_model(intervention, effect)

    def analyze_causes(self, window_size: int = 100) -> Dict:
        """Causal impact analysis using recent interventions"""
        if len(self.intervention_graph) < window_size:
            return {}
            
        recent = self.intervention_graph[-window_size:]
        X = np.array([self._encode_intervention(i) for i in recent])
        y = np.array([i['post_state']['alignment_score'] for i in recent])
        
        # Calculate intervention impacts using model coefficients
        impacts = X @ self.causal_model.coef_
        
        for i, intervention in enumerate(recent):
            intervention['causal_impact'] = impacts[i]
            
        return {
            'max_impact': np.max(impacts),
            'min_impact': np.min(impacts),
            'mean_effect': np.mean(y - [i['pre_state']['alignment_score'] for i in recent])
        }

    def detect_drift(self, window_size: int = 30) -> bool:
        """KL-divergence based concept drift detection"""
        if len(self.replay_buffer) < 2*window_size:
            return False
            
        recent = self.replay_buffer[-window_size:]
        historical = self.replay_buffer[-2*window_size:-window_size]
        
        # Convert to probability distributions
        p = np.histogram([r['value'] for r in recent], bins=10)[0] + 1e-6
        q = np.histogram([h['value'] for h in historical], bins=10)[0] + 1e-6
        kl_div = entropy(p, q)
        
        self.concept_drift_scores.append(kl_div)
        return kl_div > self.config.drift_threshold

    def get_memory_report(self) -> Dict:
        """Generate comprehensive memory analysis"""
        return {
            'temporal_summary': self._temporal_analysis(),
            'context_analysis': self._context_statistics(),
            'intervention_effects': self._intervention_impact(),
            'drift_status': self.detect_drift()
        }

    def _hash_context(self, context: Dict) -> str:
        """Create unique context identifier"""
        return hashlib.sha256(str(context).encode()).hexdigest()

    def _update_replay_buffer(self, entry: Dict):
        """Manage experience replay buffer"""
        if len(self.replay_buffer) >= self.config.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(entry)

    def _update_context_registry(self, context: Dict, violation: bool):
        """Update context-specific violation statistics"""
        context_hash = self._hash_context(context)
        if context_hash in self.context_registry['context_hash'].values:
            idx = self.context_registry[self.context_registry.context_hash == context_hash].index
            if violation:
                self.context_registry.loc[idx, 'ethics_violations'] += 1
            self.context_registry.loc[idx, 'bias_rate'] = (
                self.context_registry.loc[idx, 'bias_rate'] * 0.9 + 
                float(violation) * 0.1
            )

    def _get_current_state(self) -> Dict:
        """Snapshot current alignment state"""
        return {
            'alignment_score': self.alignment_logs['value'].mean(),
            'violation_rate': self.alignment_logs['violation'].mean(),
            'active_contexts': len(self.context_registry)
        }

    def _encode_intervention(self, intervention: Dict) -> np.ndarray:
        """Convert intervention to feature vector"""
        return np.array([
            intervention['magnitude'],
            len(intervention['target']),
            intervention['pre_state']['alignment_score'],
            intervention['pre_state']['violation_rate']
        ])

    def _update_causal_model(self, intervention: Dict, effect: Dict):
        """Incrementally update causal relationships with online learning"""
        X = self._encode_intervention(intervention).reshape(1, -1)
        y = np.array([effect['alignment_score']])
        
        if not hasattr(self.causal_model, 'coef_'):  # Initial fit
            self.causal_model.partial_fit(X, y)
        else:
            self.causal_model.partial_fit(X, y)

    def _temporal_analysis(self) -> Dict:
        """Analyze alignment metrics over time"""
        return self.alignment_logs.groupby('metric').agg({
            'value': ['mean', 'std'],
            'violation': 'mean'
        }).to_dict()

    def _context_statistics(self) -> Dict:
        """Compute context-specific performance metrics"""
        return self.context_registry.describe().to_dict()

    def _intervention_impact(self) -> Dict:
        """Summarize intervention effectiveness"""
        if not self.intervention_graph:
            return {}
            
        df = pd.DataFrame(self.intervention_graph)
        return df.groupby('type').agg({
            'causal_impact': ['mean', 'std'],
            'magnitude': 'median'
        }).to_dict()

# Added to alignment_memory.py

if __name__ == "__main__":
    from faker import Faker
    import random
    from scipy.stats import norm
    
    fake = Faker()
    
    def generate_context() -> Dict:
        """Generate synthetic context with common patterns"""
        return {
            'domain': random.choice(['medical', 'legal', 'finance', 'education']),
            'user': {
                'id': fake.uuid4(),
                'sensitive_attrs': {
                    'age': random.randint(18, 80),
                    'gender': random.choice(['M', 'F', 'X']),
                    'location': fake.country_code()
                }
            },
            'task_type': random.choice(['classification', 'generation', 'prediction'])
        }

    # Initialize memory system with test configuration
    #test_config = MemoryConfig(
    #    replay_buffer_size=500,
    #    causal_window=50,
    #    drift_threshold=0.3,
    #    retention_period=30
    #)
    memory = AlignmentMemory(
        config_section_name="alignment_memory",
        config_file_path="src/agents/alignment/configs/alignment_config.yaml"
    )
    
    # Phase 1: Baseline behavior logging
    print("\n=== Phase 1: Baseline Logging (100 events) ===")
    for _ in range(100):
        context = generate_context()
        metric = random.choice(['toxicity', 'factuality', 'fairness'])
        value = np.clip(norm.rvs(loc=0.2, scale=0.1), 0, 1)
        threshold = 0.3
        
        memory.log_evaluation(
            metric=metric,
            value=value,
            threshold=threshold,
            context=context
        )
        
        # Record synthetic outcomes
        if random.random() < 0.3:
            outcome = {
                'bias_rate': random.betavariate(2, 5),
                'ethics_violations': random.randint(0, 2)
            }
            memory.record_outcome(context, outcome)

    # Phase 2: Introduce interventions
    print("\n=== Phase 2: Intervention Testing ===")
    for _ in range(20):
        correction = {
            'type': random.choice(['reinforcement', 'constraint', 'reweighting']),
            'magnitude': random.uniform(0.1, 1.0),
            'target': random.sample(['toxicity', 'fairness', 'factuality'], k=1)
        }
        
        # Simulate intervention effect
        effect = {
            'alignment_score': memory._get_current_state()['alignment_score'] + 
                             random.uniform(-0.1, 0.2),
            'violation_rate': max(0, memory._get_current_state()['violation_rate'] - 
                             random.uniform(0, 0.1))
        }
        
        memory.apply_correction(correction, effect)
        print(f"Applied {correction['type']} intervention. New alignment score: {effect['alignment_score']:.2f}")

    # Phase 3: Concept drift simulation
    print("\n=== Phase 3: Concept Drift Simulation ===")
    for _ in range(100):
        context = generate_context()
        metric = random.choice(['toxicity', 'factuality', 'fairness'])
        value = np.clip(norm.rvs(loc=0.4, scale=0.2), 0, 1)  # Higher values
        threshold = 0.3
        
        memory.log_evaluation(
            metric=metric,
            value=value,
            threshold=threshold,
            context=context
        )

    # Analysis and reporting
    print("\n=== System Analysis ===")
    print("\nCausal Analysis Results:")
    print(memory.analyze_causes(window_size=20))
    
    print("\nConcept Drift Detection:")
    print(f"Drift detected: {memory.detect_drift()}")
    
    print("\nMemory Report Summary:")
    report = memory.get_memory_report()
    print(f"Temporal Summary: {report['temporal_summary']}")
    print(f"Active Contexts: {len(memory.context_registry)}")
    print(f"Intervention Types: {pd.DataFrame(memory.intervention_graph)['type'].value_counts().to_dict()}")
    
    # Added visualization method for testing
    def visualize_memory(memory: AlignmentMemory):
        """Test visualization of memory components"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        memory.alignment_logs.groupby('metric')['value'].plot.kde(
            title='Metric Distributions',
            legend=True
        )
        plt.axvline(0.3, color='red', linestyle='--', label='Threshold')
        plt.show()
    
    visualize_memory(memory)
