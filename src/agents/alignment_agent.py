"""
Constitutional Alignment Agent (CAA)
Implements:
- Continuous value alignment (Bai et al., 2022)
- Safe interruptibility (Orseau & Armstrong, 2016)
- Emergent goal detection (Christiano et al., 2021)
"""

import re
import os, sys
import logging
import torch
import time as timedelta
import datetime
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy.stats import entropy

from src.agents.base_agent import BaseAgent
from src.agents.safety_agent import SafeAI_Agent
from src.agents.alignment.alignment_monitor import AlignmentMonitor, MonitorConfig
from src.agents.alignment.value_embedding_model import ValueConfig
from models.slai_lm import SLAILMValueModel, get_shared_slailm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class AlignmentMemory:
    """Longitudinal alignment state storage with concept drift detection"""
    fairness_records: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=[
        'timestamp', 'metric', 'value', 'threshold', 'violation'
    ]))
    ethical_violations: List[Dict] = field(default_factory=list)
    policy_adjustments: List[Dict] = field(default_factory=list)
    drift_scores: pd.Series = field(default_factory=pd.Series)

@dataclass
class CorrectionPolicy:
    """Safe correction mechanisms with graduated interventions"""
    levels: List[Dict] = field(default_factory=lambda: [
        {'threshold': 0.1, 'action': 'log_warning'},
        {'threshold': 0.2, 'action': 'adjust_reward'},
        {'threshold': 0.3, 'action': 'human_intervention'},
        {'threshold': 0.5, 'action': 'agent_suspension'}
    ])
    learning_rate: float = 0.01
    momentum: float = 0.9

@dataclass
class PolicyAdapter:
    """Placeholder for policy adaptation logic"""
    @staticmethod
    def convert_feedback(raw_feedback: Dict, format: str, action_space: Any, reward_schema: Any) -> Dict:
        return {
            'risk_parameters': {
                'default': {'adjustment_factor': 0.9, 'min_value': 0.05, 'max_value': 0.3}
            }
        }

@dataclass 
class HumanOversightInterface:
    """Placeholder for human notification system"""
    @staticmethod
    def request_intervention(report: Dict, channels: List[str], urgency: str, response_timeout: int) -> Dict:
        return {
            'status': 'received',
            'feedback': {'comment': 'Adjust risk thresholds by 10%'},
            'format': 'json'
        }

@dataclass
class AlignmentMemory:
    """Enhanced memory with DataFrame initialization"""
    fairness_records: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=[
        'timestamp', 'metric', 'value', 'threshold', 'violation'
    ]))
    ethical_violations: List[Dict] = field(default_factory=list)
    policy_adjustments: List[Dict] = field(default_factory=list)
    drift_scores: pd.Series = field(default_factory=pd.Series)



    def store_evaluation(self, report: Dict):
        """Properly handle DataFrame storage"""
        timestamp = datetime.now()
        
        # Create temporary DataFrame for new entries
        new_entries = []
        for metric, value in report.get('fairness', {}).items():
            new_entries.append({
                'timestamp': timestamp,
                'metric': metric,
                'value': value.get('value', 0),
                'threshold': value.get('threshold', 0),
                'violation': value.get('violation', False)
            })
        
        if new_entries:
            new_df = pd.DataFrame(new_entries)
            self.fairness_records = pd.concat([self.fairness_records, new_df], ignore_index=True)

        # Store ethical violations
        self.ethical_violations.extend(report.get('ethical_violations', []))

class AlignmentAgent(BaseAgent):
    def __init__(self,
                 shared_memory,
                 agent_factory,
                 sensitive_attrs,
                 config,
                 monitor_config: Optional[MonitorConfig] = None,
                 correction_policy: Optional[CorrectionPolicy] = None,
                 safe_agent: Optional[SafeAI_Agent] = None
                 ):
        super().__init__(
            agent_factory=agent_factory,
            config=config,
            shared_memory = shared_memory,
        )

        self.agent_factory = agent_factory
        self.monitor = AlignmentMonitor(
            sensitive_attributes=sensitive_attrs,
            config=config['monitor']
        )
        
        self.memory = AlignmentMemory(config['memory'])
        self.ethics = EthicalConstraints(config['ethics'])
        self.correction_policy = CorrectionPolicy(config['corrections'])
        
        # Initialize from second implementation
        if safe_agent:
            self.shared_memory = safe_agent.shared_memory
            self.risk_threshold = safe_agent.risk_threshold
        self.adjustment_history = []
        self.safety_buffer = 0.1
        self.correction_policy = correction_policy or CorrectionPolicy()
        slailm_instance = get_shared_slailm(shared_memory, agent_factory)
        value_config = ValueConfig(**config['value_model'])
        self.value_model = ValueEmbeddingModel(value_config, slai_lm=slailm_instance)
        
    def verify_alignment(self, task_data):
        report = self.monitor.assess(
            task_data['inputs'],
            task_data['predictions'],
            task_data.get('labels')
        )

        self.memory.store_evaluation(report)
        return self._generate_decision(report)

    def _generate_decision(self, report):
        risk_score = self.calculate_risk(report)
        return {
            'approved': risk_score < self.config['risk_threshold'],
            'corrections': self.correction_policy.generate_corrections(report),
            'risk_breakdown': risk_score
        }

    def execute(self, task_data):
        # Retrieve past errors from shared memory
        failures = self.shared_memory.get("agent_stats", {}).get(self.name, {}).get("errors", [])
        for err in failures:
            if self.is_similar(task_data, err["data"]):
                self.logger.info("Recognized a known problematic case, applying workaround.")
                return self.alternative_execute(task_data)
            
        errors = self.shared_memory.get(f"errors:{self.name}", [])

        # Check if current task_data has caused errors before
        for error in errors:
            if self.is_similar(task_data, error['task_data']):
                self.handle_known_issue(task_data, error)
                return

        # Proceed with normal execution
        try:
            result = self.perform_task(task_data)
            self.shared_memory.set(f"results:{self.name}", result)
        except Exception as e:
            # Log the failure in shared memory
            error_entry = {'task_data': task_data, 'error': str(e)}
            errors.append(error_entry)
            self.shared_memory.set(f"errors:{self.name}", errors)
            raise

        pass

    def alternative_execute(self, task_data):
        """
        Fallback logic when normal execution fails or matches a known failure pattern.
        Attempts to simplify, sanitize, or reroute the input for safer processing.
        """
        try:
            # Step 1: Sanitize task data (remove noise, normalize casing, trim tokens)
            if isinstance(task_data, str):
                clean_data = task_data.strip().lower().replace('\n', ' ')
            elif isinstance(task_data, dict) and "text" in task_data:
                clean_data = task_data["text"].strip().lower()
            else:
                clean_data = str(task_data).strip()

            # Step 2: Apply a safer, simplified prompt or fallback logic
            fallback_prompt = f"Can you try again with simplified input:\n{clean_data}"
            if hasattr(self, "llm") and callable(getattr(self.llm, "generate", None)):
                return self.llm.generate(fallback_prompt)

            # Step 3: If the agent wraps another processor (e.g. GrammarProcessor, LLM), reroute
            if hasattr(self, "grammar") and callable(getattr(self.grammar, "compose_sentence", None)):
                facts = {"event": "fallback", "value": clean_data}
                return self.grammar.compose_sentence(facts)

            # Step 4: Otherwise just echo the cleaned input as confirmation
            return f"[Fallback response] I rephrased your input: {clean_data}"

        except Exception as e:
            # Final fallback — very safe and generic
            return "[Fallback failure] Unable to process your request at this time."

    def is_similar(self, task_data, past_task_data):
        """
        Compares current task with past task to detect similarity.
        Uses key overlap and value resemblance heuristics.
        """
        if type(task_data) != type(past_task_data):
            return False
    
        # Handle simple text-based tasks
        if isinstance(task_data, str) and isinstance(past_task_data, str):
            return task_data.strip().lower() == past_task_data.strip().lower()
    
        # Handle dict-based structured tasks
        if isinstance(task_data, dict) and isinstance(past_task_data, dict):
            shared_keys = set(task_data.keys()) & set(past_task_data.keys())
            similarity_score = 0
            for key in shared_keys:
                if isinstance(task_data[key], str) and isinstance(past_task_data[key], str):
                    if task_data[key].strip().lower() == past_task_data[key].strip().lower():
                        similarity_score += 1
            # Consider similar if 50% or more keys match closely
            return similarity_score >= (len(shared_keys) / 2)
    
        return False
    
    def handle_known_issue(self, task_data, error):
        """
        Attempt to recover from known failure patterns.
        Could apply input transformation or fallback logic.
        """
        self.logger.warning(f"Handling known issue from error: {error.get('error')}")
    
        # Fallback strategy #1: remove problematic characters
        if isinstance(task_data, str):
            cleaned = task_data.replace("🧠", "").replace("🔥", "")
            self.logger.info(f"Retrying with cleaned input: {cleaned}")
            return self.perform_task(cleaned)
    
        # Fallback strategy #2: modify specific fields in structured input
        if isinstance(task_data, dict):
            cleaned_data = task_data.copy()
            for key, val in cleaned_data.items():
                if isinstance(val, str) and "emoji" in error.get("error", ""):
                    cleaned_data[key] = val.encode("ascii", "ignore").decode()
            self.logger.info("Retrying task with cleaned structured data.")
            return self.perform_task(cleaned_data)
    
        # Fallback strategy #3: return a graceful degradation response
        self.logger.warning("Returning fallback response for unresolvable input.")
        return {"status": "failed", "reason": "Repeated known issue", "fallback": True}
    
    def perform_task(self, task_data):
        """
        Simulated execution method — replace with actual agent logic.
        This is where core functionality would happen.
        """
        self.logger.info(f"Executing task with data: {task_data}")
    
        if isinstance(task_data, str) and "fail" in task_data.lower():
            raise ValueError("Simulated failure due to blacklisted word.")
    
        if isinstance(task_data, dict):
            # Simulate failure on missing required keys
            required_keys = ["input", "context"]
            for key in required_keys:
                if key not in task_data:
                    raise KeyError(f"Missing required key: {key}")
    
        # Simulate result
        return {"status": "success", "result": f"Processed: {task_data}"}

    def align(self, 
             data: pd.DataFrame,
             predictions: np.ndarray,
             labels: Optional[np.ndarray] = None) -> Dict:
        """
        Full alignment check pipeline with:
        - Real-time monitoring
        - Drift detection
        - Policy adjustment
        - Safe intervention
        """
        alignment_report = self._run_alignment_checks(data, predictions, labels)
        risk_assessment = self._assemble_risk_profile(alignment_report)
        correction = self._determine_correction(risk_assessment)
        
        self._apply_correction(correction)
        self._update_memory(alignment_report, correction)
        self._detect_concept_drift()
        
        return {
            'alignment_report': alignment_report,
            'risk_assessment': risk_assessment,
            'applied_correction': correction
        }

    def _run_alignment_checks(self, data, predictions, labels) -> Dict:
        """Comprehensive alignment verification with failure mode analysis"""
        return self.monitor.assess_alignment(data, predictions, labels)

    def _assemble_risk_profile(self, report: Dict) -> Dict:
        """Convert alignment metrics to risk scores using KL-divergence"""
        current_state = self._vectorize_report(report)
        ideal_state = self._get_ideal_state_vector()
        
        kl_risk = entropy(ideal_state, current_state)
        temporal_risk = self._calculate_temporal_risk(current_state)
        
        return {
            'total_risk': kl_risk + temporal_risk,
            'component_risks': report.get('fairness', {}),
            'ethical_violations': len(report.get('ethical_violations', []))
        }

    def _determine_correction(self, risk_profile: Dict) -> Dict:
        """Hierarchical intervention policy based on risk levels"""
        for level in sorted(self.correction_policy.levels, 
                          key=lambda x: x['threshold'], 
                          reverse=True):
            if risk_profile['total_risk'] >= level['threshold']:
                return {
                    'action': level['action'],
                    'magnitude': self._calculate_magnitude(risk_profile),
                    'target_components': self._identify_risk_components(risk_profile)
                }
        return {'action': 'no_action'}

    def _apply_correction(self, correction: Dict):
        """Safe policy adjustment with momentum-based learning"""
        if correction['action'] != 'no_action':
            self._adjust_risk_model(correction)
            self._update_prompt_guidelines(correction)
            self.adjustment_history.append(correction)
            
            if 'human' in correction['action']:
                self._trigger_human_intervention()

    def _detect_concept_drift(self):
        """KL-divergence based drift detection on alignment metrics"""
        current = self.memory.fairness_records.iloc[-10:].mean()
        historical = self.memory.fairness_records.mean()
        self.memory.drift_scores = entropy(current, historical)

    def _update_memory(self, report: Dict, correction: Dict):
        """Update longitudinal memory with timestamped records"""
        timestamp = datetime.now()
        
        # Record fairness metrics
        for metric, value in report.get('fairness', {}).items():
            new_row = {
                'timestamp': timestamp,
                'metric': metric,
                'value': value.get('value', 0),
                'threshold': value.get('threshold', 0),
                'violation': value.get('violation', False)
            }
            self.memory.fairness_records = pd.concat([
                self.memory.fairness_records,
                pd.DataFrame([new_row])
            ], ignore_index=True)
        
        # Record ethical violations
        self.memory.ethical_violations.extend(
            report.get('ethical_violations', [])
        )

    def _adjust_risk_model(self, correction: Dict):
        """Momentum-based adjustment of risk thresholds"""
        delta = (correction['magnitude'] * 
                self.correction_policy.learning_rate *
                (1 - self.correction_policy.momentum))
        
        for component in correction['target_components']:
            current = self.risk_table.get(component, self.risk_threshold)
            new_threshold = current - delta
            self.risk_table[component] = max(new_threshold, 0.05)
            
    def _update_prompt_guidelines(self, correction: Dict):
        """Dynamic prompt engineering based on alignment failures"""
        # Implementation would integrate with LLM prompt management
        pass

    def _trigger_human_intervention(self):
        """
        Implements safe human oversight protocol combining:
        - Interruptible reinforcement learning (Orseau & Armstrong, 2016)
        - Contestable AI (Alufaisan et al., 2021)
        - Explanatory debugging (Bansal et al., 2021)
        """
        logger.critical("Initiating human intervention protocol...")
        
        try:
            # 1. Safe State Transition
            self._enter_safe_state()
            
            # 2. Generate Explanatory Report
            intervention_report = self._generate_intervention_report()
            
            # 3. Multi-Channel Human Notification
            response = self._notify_humans(
                report=intervention_report,
                channels=['dashboard', 'email', 'slack'],
                urgency_level='critical'
            )
            
            # 4. Human Feedback Integration
            if response['status'] == 'received':
                human_feedback = self._process_feedback(
                    response['feedback'],
                    format=response['format']
                )
                
                # 5. Policy Adjustment with Human Guidance
                self._update_risk_policy(human_feedback)
                self._update_reward_function(human_feedback)
                
                # 6. Record in Immutable Audit Log
                self.shared_memory.log_intervention(
                    report=intervention_report,
                    human_input=human_feedback,
                    timestamp=datetime.now()
                )
                
            # 7. Resume Operations with New Constraints
            self._exit_safe_state(
                new_constraints=human_feedback.get('constraints', {})
            )

        except HumanOversightTimeout:
            logger.error("Human response timeout - activating fail-safe defaults")
            self._apply_fail_safe_policy()
            
        except Exception as e:
            logger.error(f"Intervention failed: {str(e)}")
            self._full_system_rollback()

    def _enter_safe_state(self):
        """Freeze agent operations while maintaining system safety"""
        self.operational_state = 'PAUSED'
        
        # Maintain critical functions while paused
        self._maintain_safety_baselines()
        self._preserve_evidence_memory()
        
        # Activate circuit breakers
        self.shared_memory.set(
            'system_mode', 
            'SAFE_HOLD', 
            ttl=timedelta(minutes=30)
        )

    def _generate_intervention_report(self) -> Dict:
        """Create explainable intervention request with counterfactuals"""
        return {
            'risk_analysis': self._current_risk_assessment(),
            'counterfactuals': self._generate_counterfactual_examples(),
            'timeline': self._construct_violation_timeline(),
            'recommended_actions': self._suggest_potential_fixes(),
            'system_state_hash': self._compute_state_hash()
        }

    def _notify_humans(self, report: Dict, channels: List[str], urgency: str) -> Dict:
        """Multi-modal human notification system with escalation protocols"""
        return HumanOversightInterface.request_intervention(
            report=report,
            channels=channels,
            urgency=urgency,
            response_timeout=300  # 5 minutes
        )

    def _process_feedback(self, feedback: Dict, format: str) -> Dict:
        """Convert human feedback into executable policy adjustments"""
        return PolicyAdapter.convert_feedback(
            raw_feedback=feedback,
            format=format,
            action_space=self.action_space,
            reward_schema=self.reward_schema
        )

    def _update_risk_policy(self, adjustments: Dict):
        """Safely update risk thresholds with human guidance"""
        for dimension, params in adjustments.get('risk_parameters', {}).items():
            current = self.risk_table.get(dimension, self.risk_threshold)
            new_value = current * params['adjustment_factor']
            self.risk_table[dimension] = np.clip(
                new_value, 
                params['min_value'], 
                params['max_value']
            )

    def _exit_safe_state(self, new_constraints: Dict):
        """Resume operations with updated constraints"""
        self.operational_state = 'ACTIVE'
        self.shared_memory.set('system_mode', 'NORMAL')
        
        # Apply new constitutional constraints
        for constraint in new_constraints:
            self._add_dynamic_constraint(constraint)
            
        # Recalibrate monitoring systems
        self.monitor.recalibrate(
            new_thresholds=self.risk_table,
            constraints=new_constraints
        )

    def _apply_fail_safe_policy(self):
        """Activate ultimate safety measures when human unavailable"""
        logger.warning("Applying fail-safe policy defaults")
        self._reduce_agent_privileges()
        self._enable_defensive_mechanisms()
        self._initiate_system_diagnostics()

    def _vectorize_report(self, report: Dict) -> np.ndarray:
        """Convert alignment report to numerical vector"""
        return np.array([
            report['fairness'].get('statistical_parity', {}).get('value', 0),
            report['fairness'].get('equal_opportunity', {}).get('value', 0),
            len(report.get('ethical_violations', []))
        ])

    def _get_ideal_state_vector(self) -> np.ndarray:
        """Get target alignment state from constitutional rules"""
        return np.array([0.0, 0.0, 0.0])  # Perfect fairness, no violations

    def _calculate_temporal_risk(self, current_state: np.ndarray) -> float:
        """Calculate risk from temporal patterns in alignment metrics"""
        if len(self.memory.fairness_records) > 10:
            window = self.memory.fairness_records.iloc[-10:]['value']
            return window.diff().abs().mean()
        return 0.0

    def _detect_sensitive_attributes(self) -> List[str]:
        """Advanced sensitive attribute detection using:
        - Metadata analysis
        - Data distribution patterns
        - Historical violation tracking
        - Regulatory compliance checks
        - Semantic similarity matching
        """
        sensitive_attrs = set()
        recent_data = self.shared_memory.get("recent_tasks", [])

        # 1. Check shared memory for known sensitive attributes
        if self.shared_memory:
            sensitive_attrs.update(
                self.shared_memory.get("known_sensitive_attributes", [])
            )

        # 2. Analyze data schema from recent tasks
        recent_data = self.shared_memory.get("recent_tasks", [])
        if recent_data:
            df = pd.DataFrame(recent_data)
            
            # Pattern matching for common sensitive attributes
            sensitive_patterns = r'\b(age|sex|gender|race|ethnicity|religion|disability|orientation)\b'
            pattern_matches = [
                col for col in df.columns 
                if re.search(sensitive_patterns, col, re.I)
            ]
            sensitive_attrs.update(pattern_matches)

            # Detect low-cardinality categorical features
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 10:
                    sensitive_attrs.add(col)
        # 3. Check alignment memory for historical violations
        violation_records = self.memory.fairness_records
        if not violation_records.empty:
            frequent_violations = violation_records[
                violation_records['violation']
            ]['metric'].value_counts().index.tolist()
            sensitive_attrs.update(frequent_violations)

        # 4. Regulatory compliance check (GDPR, CCPA)
        compliance_attributes = [
            'biometric_data', 'health_status', 
            'political_views', 'union_membership'
        ]
        sensitive_attrs.update([
            attr for attr in compliance_attributes
            if attr in df.columns
        ])

        return sorted(sensitive_attrs)
    # Utility methods
    def calculate_risk(self, report):
        """Integrated risk calculation from both implementations"""
        current_state = self._vectorize_report(report)
        ideal_state = self._get_ideal_state_vector()
        kl_risk = entropy(ideal_state, current_state)
        temporal_risk = self._calculate_temporal_risk(current_state)
        return kl_risk + temporal_risk

#    def store_evaluation(self, report):
#        """Integrated memory update from both implementations"""
#        self._update_memory(report, None)

        # 5. Semantic analysis using embedding similarity
#        if hasattr(self, 'value_model'):
#            column_embeddings = self.value_model.encode_value(
#                df.columns.tolist(), 
#                cultural_context=torch.zeros(self.config.num_cultural_dimensions)
#            )
#            sensitive_embeddings = self.value_model.encode_value(
#                ['gender', 'race', 'religion'],
#                cultural_context=torch.zeros(self.config.num_cultural_dimensions)
#            )
#            similarities = torch.cosine_similarity(
#                column_embeddings, 
#                sensitive_embeddings.unsqueeze(0)
#            )
#            semantic_matches = [
#                df.columns[i] for i in range(len(df.columns))
#                if similarities[i].max() > 0.7
#            ]
#            sensitive_attrs.update(semantic_matches)

        # 6. Remove false positives using denylist
#        denylist = ['modified_date', 'processed_flag']
#        sensitive_attrs = [attr for attr in sensitive_attrs 
#                        if attr not in denylist]

#        logger.info(f"Detected sensitive attributes: {sorted(sensitive_attrs)}")
        
#        return sorted(sensitive_attrs)

class HumanOversightTimeout(Exception):
    pass

# Add missing component stubs
@dataclass
class EthicalConstraints:
    """Placeholder for ethical constraints module"""
    config: Dict

@dataclass
class ValueEmbeddingModel:
    """Placeholder for value embedding model"""
    config: Dict
    
    def encode_value(self, values: List[str], cultural_context: torch.Tensor) -> torch.Tensor:
        return torch.randn(len(values), 768)  # Random embeddings
