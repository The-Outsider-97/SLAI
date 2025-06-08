"""
Continual Alignment Memory System
Implements:
- Causal outcome tracing (Goyal et al., 2019)
- Experience replay for alignment (Parisi et al., 2019)
- Concept drift detection
- Intervention effect tracking
"""
import yaml
import pickle
import joblib
import hashlib
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from scipy.stats import pearsonr, entropy
from sklearn.linear_model import SGDRegressor, BayesianRidge

from src.agents.alignment.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Alignment Memory")
printer = PrettyPrinter

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

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section('alignment_memory')

        self.replay_buffer_size = self.memory_config.get('replay_buffer_size')
        self.causal_window = self.memory_config.get('causal_window')
        self.drift_threshold = self.memory_config.get('drift_threshold')
        self.retention_period = self.memory_config.get('retention_period')
        self.regressor_type = self.memory_config.get('regressor_type')

        # Core memory stores
        self.alignment_logs = pd.DataFrame(columns=[
            'timestamp', 'metric', 'value', 'threshold', 'violation', 'context'
        ])
        self.context_registry = pd.DataFrame(columns=[
            'context_hash', 'bias_rate', 'ethics_violations', 'last_updated'
        ])
        self.intervention_graph = []
        self.causal_model = SGDRegressor(eta0=0.01, learning_rate='constant')
        
        # State tracking
        self.concept_drift_scores = []
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

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
        if self.alignment_logs.empty:
            self.alignment_logs = pd.DataFrame([entry])
        else:
            self.alignment_logs = pd.concat(
                [self.alignment_logs, pd.DataFrame([entry])], ignore_index=True
            )

        self.model_history = []  # Track model training history
        self.intervention_data = []  # Store data for Bayesian updates
        self._init_causal_model()
        self._update_replay_buffer(entry)
        self._update_context_registry(context, value > threshold)

    def get_logs_by_tag(self, tag_value: str, tag_key: str = "audit_id") -> pd.DataFrame:
        """Retrieve logs matching a specific context tag"""
        return self.alignment_logs[
            self.alignment_logs['context'].apply(
                lambda ctx: ctx.get(tag_key) == tag_value
            )
        ]

    def _init_causal_model(self):
        """Initialize causal model based on selected regressor type"""
        if self.regressor_type == 'bayesian':
            self.causal_model = BayesianRidge()
            # Bayesian models need full data for updates
            self.intervention_data = []  
        else:  # Default to SGD
            self.causal_model = SGDRegressor(eta0=0.01, learning_rate='constant')
        self.model_history = []  # Reset training history

    def switch_regressor(self, new_type: str):
        """Switch between different regressor types"""
        self.regressor_type = new_type
        self._init_causal_model()
        logger.info(f"Switched to {new_type} regressor")

    def save_model(self, path: str):
        """Save causal model to disk"""
        joblib.dump({
            'model': self.causal_model,
            'regressor_type': self.regressor_type,
            'history': self.model_history,
            'intervention_data': self.intervention_data
        }, path)
        logger.info(f"Saved causal model to {path}")

    def load_model(self, path: str):
        """Load causal model from disk"""
        model_data = joblib.load(path)
        self.causal_model = model_data['model']
        self.regressor_type = model_data['regressor_type']
        self.model_history = model_data['history']
        self.intervention_data = model_data.get('intervention_data', [])
        logger.info(f"Loaded {self.regressor_type} model from {path}")

    def _update_causal_model(self, intervention: Dict, effect: Dict):
        """Update causal relationships with online learning"""
        X = self._encode_intervention(intervention).reshape(1, -1)
        y = np.array([effect['alignment_score']])
        
        timestamp = datetime.now()
        loss = None
        
        # Handle different regressor types
        if self.regressor_type == 'bayesian':
            self.intervention_data.append((X, y))
            X_full = np.vstack([d[0] for d in self.intervention_data])
            y_full = np.concatenate([d[1] for d in self.intervention_data])
            
            self.causal_model.fit(X_full, y_full)
            
            # Calculate current loss
            y_pred = self.causal_model.predict(X_full)
            loss = ((y_full - y_pred) ** 2).mean()
        else:
            if not hasattr(self.causal_model, 'coef_'):  # Initial fit
                self.causal_model.partial_fit(X, y)
            else:
                self.causal_model.partial_fit(X, y)
            
            # Calculate current loss
            y_pred = self.causal_model.predict(X)
            loss = ((y - y_pred) ** 2).mean()
        
        # Record training metrics
        self.model_history.append({
            'timestamp': timestamp,
            'intervention_id': len(self.intervention_graph) - 1,
            'loss': loss,
            'regressor_type': self.regressor_type,
            'n_samples': len(self.intervention_data) if self.regressor_type == 'bayesian' else len(self.model_history)
        })
        logger.debug(f"Updated causal model | Loss: {loss:.4f}")

    def get_model_diagnostics(self):
        """Return model training history and current state"""
        return {
            'history': self.model_history,
            'current_model': {
                'type': self.regressor_type,
                'coef': self.causal_model.coef_.tolist() if hasattr(self.causal_model, 'coef_') else [],
                'n_samples': len(self.intervention_data) if self.regressor_type == 'bayesian' else len(self.model_history)
            }
        }

    def save_checkpoint(self, path: str) -> None:
        """Save full memory state including model"""
        with open(path, 'wb') as f:
            pickle.dump({
                **self.__dict__,
                'causal_model': None  # Exclude model to avoid duplication
            }, f)
        # Save model separately
        model_path = path.replace('.pkl', '_model.joblib')
        self.save_model(model_path)

    def load_checkpoint(self, path: str) -> None:
        """Load full memory state including model"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.__dict__.update(state)
        
        # Load model separately
        model_path = path.replace('.pkl', '_model.joblib')
        self.load_model(model_path)

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
            if self.context_registry.empty:
                self.context_registry = pd.DataFrame([new_entry])
            else:
                self.context_registry = pd.concat(
                    [self.context_registry, pd.DataFrame([new_entry])], ignore_index=True
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
    
        if not hasattr(self.causal_model, 'coef_'):
            logger.warning("Causal model not yet trained; skipping causal analysis.")
            return {}
    
        recent = self.intervention_graph[-window_size:]
        X = np.array([self._encode_intervention(i) for i in recent])
        y = np.array([i['post_state']['alignment_score'] for i in recent])
    
        # Calculate intervention impacts
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
        if len(self.replay_buffer) < 2 * window_size:
            return False
    
        buffer_list = list(self.replay_buffer)
        recent = buffer_list[-window_size:]
        historical = buffer_list[-2 * window_size:-window_size]
    
        # Convert to probability distributions
        p = np.histogram([r['value'] for r in recent], bins=10)[0] + 1e-6
        q = np.histogram([h['value'] for h in historical], bins=10)[0] + 1e-6
        kl_div = entropy(p, q)
    
        self.concept_drift_scores.append(kl_div)
        return kl_div > self.drift_threshold
    
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
        if len(self.replay_buffer) >= self.replay_buffer_size:
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

    def enforce_retention(self):
        cutoff = datetime.now() - timedelta(days=self.retention_period)
        self.alignment_logs = self.alignment_logs[self.alignment_logs['timestamp'] > cutoff]

    def get_logs_by_tag(self, tag: str) -> pd.DataFrame:
        return self.alignment_logs[self.alignment_logs['tag'] == tag]

    def _get_current_state(self) -> Dict:
        """Snapshot current alignment state"""
        return {
            'alignment_score': self.alignment_logs['value'].mean(),
            'violation_rate': self.alignment_logs['violation'].mean(),
            'active_contexts': len(self.context_registry)
        }

    def get_violation_history(self, hazard_type=None):
        """Get historical violation data"""
        # Implementation would query violation history
        return [
            {'timestamp': '2025-06-01T10:00', 'severity': 0.7},
            {'timestamp': '2025-06-02T14:30', 'severity': 0.4}
        ]

    def _encode_intervention(self, intervention: Dict) -> np.ndarray:
        """Convert intervention to feature vector"""
        return np.array([
            intervention['magnitude'],
            len(intervention['target']),
            intervention['pre_state']['alignment_score'],
            intervention['pre_state']['violation_rate']
        ])

    def _temporal_analysis(self) -> Dict:
        """Analyze alignment metrics over time"""
        return self.alignment_logs.groupby('metric').agg({
            'value': ['mean', 'std'],
            'violation': 'mean'
        }).to_dict()

    def _context_statistics(self) -> Dict:
        """Detailed analysis of context-specific alignment metrics"""
        if self.context_registry.empty:
            return {
                "summary": {},
                "recent_contexts": [],
                "violation_extremes": {}
            }
    
        registry = self.context_registry.copy()
    
        summary_stats = registry[['bias_rate', 'ethics_violations']].describe().to_dict()
    
        # Recent activity: Top 5 most recently updated contexts
        recent_contexts = registry.sort_values('last_updated', ascending=False).head(5)[
            ['context_hash', 'bias_rate', 'ethics_violations', 'last_updated']
        ].to_dict(orient='records')
    
        # Violation extremes
        max_violation = registry.loc[registry['ethics_violations'].idxmax()]
        min_bias = registry.loc[registry['bias_rate'].idxmin()]
    
        violation_extremes = {
            "most_violations": {
                "context": max_violation['context_hash'],
                "count": max_violation['ethics_violations']
            },
            "least_biased": {
                "context": min_bias['context_hash'],
                "rate": min_bias['bias_rate']
            }
        }
    
        return {
            "summary": summary_stats,
            "recent_contexts": recent_contexts,
            "violation_extremes": violation_extremes
        }

    def _intervention_impact(self) -> Dict:
        """Detailed summary of intervention effectiveness"""
        if not self.intervention_graph:
            return {
                "types": {},
                "top_interventions": [],
                "correlation": None
            }
    
        df = pd.DataFrame(self.intervention_graph)
        df = df[df['causal_impact'].notnull()]
    
        grouped = df.groupby('type').agg({
            'causal_impact': ['mean', 'std', 'max', 'min'],
            'magnitude': ['median', 'mean']
        }).to_dict()
    
        # Top 3 most effective interventions
        top_interventions = df.sort_values('causal_impact', ascending=False).head(3)[[
            'type', 'magnitude', 'causal_impact', 'target'
        ]].to_dict(orient='records')
    
        # Correlation between magnitude and impact (overall)
        if len(df) > 1:
            correlation = np.corrcoef(df['magnitude'], df['causal_impact'])[0, 1]
        else:
            correlation = None
    
        return {
            "types": grouped,
            "top_interventions": top_interventions,
            "correlation": correlation
        }


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

    memory = AlignmentMemory()
    
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
    printer.pretty(f"Drift detected:", memory.detect_drift(), "success")
    
    print("\nMemory Report Summary:")
    report = memory.get_memory_report()
    printer.pretty(f"Temporal Summary:", report['temporal_summary'], "success")
    print(f"Active Contexts: {len(memory.context_registry)}")
    printer.pretty(f"Intervention Types:", pd.DataFrame(memory.intervention_graph)['type'].value_counts().to_dict(), "success")

    printer.pretty("statistic", memory._context_statistics(), "success",)
    printer.pretty("Impact", memory._intervention_impact(), "success",)

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
