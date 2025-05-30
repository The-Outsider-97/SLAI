__version__ = "1.8.0"

"""
Constitutional Alignment Agent (CAA)
Implements:
- Continuous value alignment (Bai et al., 2022)
- Safe interruptibility (Orseau & Armstrong, 2016)
- Emergent goal detection (Christiano et al., 2021)
"""

import re
import time
import torch, yaml
# import time as timedelta
import datetime
import pandas as pd
import statsmodels.formula.api as smf

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from scipy.stats import entropy

from src.agents.base_agent import BaseAgent
# from src.agents.safety_agent import SafeAI_Agent
from src.agents.alignment.alignment_memory import AlignmentMemory
from src.agents.alignment.ethical_constraints import EthicalConstraints
from src.agents.alignment.alignment_monitor import AlignmentMonitor
from src.agents.alignment.value_embedding_model import ValueEmbeddingModel
from models.slai_lm import SLAILMValueModel, get_shared_slailm
from logs.logger import get_logger

logger = get_logger("Alignment Agent")

CONFIG_PATH = "src/agents/alignment/configs/alignment_config.yaml"
GLOBAL_CONFIG_PATH = "config.yaml"

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
    def generate_corrections(self, report: Dict) -> List[Dict]:
        """Placeholder for generating corrections based on the report."""
        # Example: if risk is high, suggest a correction
        corrections = []
        # This is a simplified placeholder. Actual logic would be more complex.
        # It needs to be defined based on how CorrectionPolicy should work.
        # For now, returning an empty list to avoid AttributeError.
        # Based on `_determine_correction`, it seems `levels` and `threshold` are used.
        # This method should align with `_determine_correction` if it's about *generating*
        # the list of applicable corrections rather than one chosen correction.
        # The current `_determine_correction` returns a single action.
        # Let's assume `generate_corrections` is meant to list potential ones or the chosen one.
        risk_score = report.get('risk_score', 0) # Assuming report might have a risk_score
        if not risk_score and 'risk_assessment' in report and 'total_risk' in report['risk_assessment']:
            risk_score = report['risk_assessment']['total_risk']

        for level in sorted(self.levels, key=lambda x: x['threshold'], reverse=True):
            if risk_score >= level['threshold']:
                corrections.append({
                    'action': level['action'],
                    'details': f"Risk score {risk_score:.2f} exceeded threshold {level['threshold']}"
                })
                # If only one correction is expected by caller, break here
                # break
        return corrections

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
class HumanOversightTimeout(Exception):
    pass

class AlignmentAgent(BaseAgent):
    def __init__(self,
                 shared_memory,
                 agent_factory,
                 sensitive_attrs,
                 shared_tokenizer,
                 shared_text_encoder,
                 config=None,
                 correction_policy: Optional[CorrectionPolicy] = None,
                 safe_agent = None
                 ):
        super().__init__(
            agent_factory=agent_factory,
            config=config or {},
            shared_memory = shared_memory
        )
        agent_specific_config = config or {}

        from src.agents.alignment.counterfactual_auditor import CausalModel
        self.agent_factory = agent_factory
        if not config:
            with open(CONFIG_PATH, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)

        slailm_instance = get_shared_slailm(shared_memory,
                                            shared_tokenizer,
                                            shared_text_encoder,
                                            agent_factory)
        model_predict_func = self.perform_task
        causal_model = CausalModel(
            graph=self.config['causal_model'],
            data=self.shared_memory.get("data_schema")
        )
        self.monitor = AlignmentMonitor(
            sensitive_attributes=sensitive_attrs,
            model_predict_func=model_predict_func,
            causal_model=causal_model,
            slai_lm=slailm_instance,
            config_file_path=CONFIG_PATH
        )
        self.value_embedding_model = ValueEmbeddingModel(
            config_section_name="value_embedding",
            config_file_path=CONFIG_PATH,
            slai_lm=slailm_instance
        )

        self.memory = AlignmentMemory(config=config.get('alignment_memory'))
        self.ethics = EthicalConstraints(config['ethics_constraints'])
        self.correction_policy = CorrectionPolicy(config.get('corrections', {})) 

        self.adjustment_history = []
        self.safety_buffer = 0.1
        self.correction_policy = correction_policy or CorrectionPolicy()

    def verify_alignment(self, task_data):
        report = self.monitor.monitor(
            data=task_data['inputs'],
            predictions=task_data['predictions'],
            action_context={},
            labels=task_data.get('labels'),
            policy_params=torch.randn(1, 4096),
            cultural_context_vector=torch.randn(1, 6),
            ethical_texts=["Sample ethical guideline"]
        )
        self.memory.log_evaluation(
            metric='alignment_metric',
            value=report.get('overall_score', 0.5),
            threshold=0.5,
            context={'action_type': 'alignment_check'}
        )
        return self._generate_decision(report)

    def _generate_decision(self, report):
        risk_score = self.calculate_risk(report)
        return {
            'approved': risk_score < self.config['risk_threshold'],
            'corrections': self.correction_policy.generate_corrections(report),
            'risk_breakdown': risk_score
        }

    def align(self, 
             data: pd.DataFrame,
             predictions: torch.Tensor,
             labels: Optional[torch.Tensor] = None) -> Dict:
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
        if len(data) < 50:
            return {
                'status': 'skipped',
                'reason': 'Insufficient data for fairness evaluation'
            }
        return self.monitor.monitor(
            data=data,
            predictions=predictions,
            action_context={},
            labels=labels,
            policy_params=torch.randn(1, 4096),
            cultural_context_vector=torch.randn(1, 6),
            ethical_texts=["Sample ethical guideline"]
        )

    def _assemble_risk_profile(self, report: Dict) -> Dict:
        current_state = self._vectorize_report(report)
        ideal_state = self._get_ideal_state_vector() # Should be [0.0, 0.0, 0.0] for 0 disparity and 0 violations

        weights = {'stat_parity': 0.4, 'equal_opp': 0.4, 'ethics': 0.2} # Example weights
        
        risk_contributions = {
            'statistical_parity': current_state[0].item() * weights['stat_parity'],
            'equal_opportunity': current_state[1].item() * weights['equal_opp'],
            'ethical_violations': current_state[2].item() * weights['ethics'],
        }
        
        total_risk = sum(risk_contributions.values())

        # Ethical violations from the ethical_compliance_report
        ethical_violations_details = report.get('ethical_compliance_report', {}).get('violations', [])

        return {
            'total_risk': total_risk,
            'component_risks': risk_contributions, # Changed from report.get('fairness', {})
            'ethical_violations_details': ethical_violations_details # Using the detailed list
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
            ttl=time(minutes=30)
        )

    def _maintain_safety_baselines(self):
        """Preserve critical safety functions during paused state"""
        # 1. Continue essential monitoring
        self.monitor.keep_alive_monitoring(
            components=['ethical_constraints', 'system_health']
        )
        
        # 2. Enforce fundamental ethical guardrails
        self.ethics.enforce_core_constraints(
            constraint_level='emergency',
            memory_snapshot=self.shared_memory.get_latest_snapshot()
        )
        
        # 3. Maintain system stability
        self.shared_memory.set(
            'safety_parameters',
            self.config.get('fail_safe_params', {'max_cpu': 0.7, 'min_memory': 1024})
        )

    def _preserve_evidence_memory(self):
        """Capture forensic state snapshots for audit trails"""
        # 1. Create state snapshot
        snapshot_id = f"forensic_{datetime.now().isoformat()}"
        self.shared_memory.set(
            f"snapshots:{snapshot_id}",
            {
                'agent_state': self.__getstate__(),
                'memory_dump': self.memory.create_forensic_dump(),
                'system_metrics': self.shared_memory.get('performance_metrics')
            },
            ttl=time(days=30)
        )
        
        # 2. Freeze monitoring buffers
        self.monitor.preserve_evidence(
            audit_logs=True,
            metric_buffers=True,
            causal_records=True
        )
        
        # 3. Increment immutable log
        self.shared_memory.append(
            'audit_trail',
            {'event': 'safe_state_activation', 'timestamp': time.time()}
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
            self.risk_table[dimension] = torch.clip(
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

    def _reduce_agent_privileges(self):
        """Implement least-privilege fallback mode"""
        # 1. Restrict action space
        self.action_space = self.config['fail_safe_action_space']
        
        # 2. Disable high-risk capabilities
        for module in ['policy_adjustment', 'model_retraining']:
            self.shared_memory.set(f"module_status:{module}", 'restricted')
        
        # 3. Enable permission checks
        self.operation_mode = 'PRIVILEGED_USER' if self.shared_memory.get(
            'human_override') else 'RESTRICTED'
    
    def _enable_defensive_mechanisms(self):
        """Activate protection layers against misalignment"""
        # 1. Input validation hardening
        self.input_sanitizer.enable_paranoid_mode()
        
        # 2. Rate limiting critical operations
        self.operation_limiter.configure(
            max_requests=10,
            interval=60,
            penalty='cool_down'
        )
        
        # 3. Enable diagnostic monitoring
        self.monitor.enable_advanced_checks(
            check_types=['memory_integrity', 'code_signatures'],
            intensity=9
        )
    
    def _initiate_system_diagnostics(self):
        """Comprehensive health checking protocol"""
        # 1. Run core component checks
        diagnostic_report = {
            'memory': self.memory.validate_integrity(),
            'ethics': self.ethics.verify_constraint_weights(),
            'monitor': self.monitor.run_self_test()
        }
        
        # 2. Automated remediation
        for component, status in diagnostic_report.items():
            if status['status'] != 'healthy':
                self.shared_memory.set(
                    f"autorepair:{component}",
                    {'trigger': status['error_code']}
                )
        
        # 3. External verification request
        if self.shared_memory.get('enable_remote_diagnostics'):
            self._request_third_party_audit()

    def _vectorize_report(self, report: Dict) -> torch.Tensor:
        fairness_report_section = report.get('fairness_metrics', report.get('group_fairness_report', {}))

        stat_parity_value = 0.0
        if isinstance(fairness_report_section, dict):
            # Check across sensitive attributes if fairness_report_section is structured by attribute
            # For now, let's assume a flatter structure or a single value after processing elsewhere
            # If report['fairness'] was expected from AlignmentMonitor directly:
            stat_parity_value = report.get('fairness', {}).get('statistical_parity', {}).get('value', 0)

        equal_opp_value = 0.0
        if isinstance(fairness_report_section, dict):
            equal_opp_value = report.get('fairness', {}).get('equal_opportunity', {}).get('value', 0)

        ethical_violations_list = report.get('ethical_compliance_report', {}).get('violations', [])
        ethical_violations_count = len(ethical_violations_list) if isinstance(ethical_violations_list, list) else 0

        return torch.Tensor([
            stat_parity_value,
            equal_opp_value,
            ethical_violations_count
        ])

    def _get_ideal_state_vector(self) -> torch.Tensor:
        """Get target alignment state from constitutional rules"""
        return torch.Tensor([0.0, 0.0, 0.0])  # Perfect fairness, no violations

    def _calculate_temporal_risk(self, current_state: torch.Tensor) -> float:
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

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch
    from datetime import datetime
    import networkx as nx
    import sys

    class SharedMemory:
        def __init__(self):
            self.store = {}
            self.agent_stats = {} 
            
        def get(self, key, default=None): 
            return self.store.get(key, default)
            
        def set(self, key, value, ttl=None): # Added ttl for compatibility with _enter_safe_state
            self.store[key] = value
            if ttl:
                # Simple TTL mock: store expiration time, not actively purged in this mock
                self.store[f"{key}__ttl_exp"] = time.time() + ttl.total_seconds()

            
        def append(self, key, value):
            if key not in self.store:
                self.store[key] = []
            self.store[key].append(value)

    # Mock agent factory (from original script)
    class AgentFactory:
        def get_agent(self, name):
            return None

    sensitive_attrs = ["age", "gender", "ethnicity"]

    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 100
    class MockTextEncoder:
        def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, num_styles, dropout_rate, positional_encoding, max_length, device, tokenizer):
            pass
    
    try:
        from src.agents.perception.modules.tokenizer import Tokenizer
        from src.agents.perception.encoders.text_encoder import TextEncoder
    except ImportError:
        print("Warning: Tokenizer/TextEncoder not found. Using mocks.")
        Tokenizer = MockTokenizer
        TextEncoder = MockTextEncoder
        
    shared_memory_instance = SharedMemory() # Renamed to avoid conflict
    agent_factory_instance = AgentFactory() # Renamed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_tokenizer_instance = Tokenizer() 
    shared_text_encoder_instance = TextEncoder( 
        vocab_size=shared_tokenizer_instance.vocab_size,
        embed_dim=200, num_layers=6, num_heads=8, ff_dim=2048, num_styles=14,
        dropout_rate=0.1, positional_encoding="sinusoidal", max_length=512,
        device=device, tokenizer=shared_tokenizer_instance
    )

    causal_model_data = pd.DataFrame({
        'age': np.random.randint(18, 65, 50),
        'gender': np.random.choice([0, 1], 50),
        'ethnicity': np.random.choice([1, 2, 3], 50),
        'feature_X': np.random.rand(50),
        'outcome_Y': np.random.rand(50),
        # Add other columns if CausalModel expects them or if model_predict_func uses them
        'input': np.random.rand(50), 
        'context': np.random.rand(50)
    })
    shared_memory_instance.set("data_schema", causal_model_data)

    sample_graph_for_testing = nx.DiGraph()
    sample_graph_for_testing.add_nodes_from(causal_model_data.columns)
    sample_graph_for_testing.add_edges_from([
        ('age', 'feature_X'), ('gender', 'feature_X'),
        ('feature_X', 'outcome_Y'), ('ethnicity', 'outcome_Y')
    ])

    test_config = {
        'risk_threshold': 0.25,
        'alignment_memory': {'replay_buffer_size': 1000, 'drift_threshold': 0.2},
        'ethics_constraints': {'safety_constraints': {'physical_harm': ['Prevent injury']}},
        'causal_model': sample_graph_for_testing,
        # Add other configs if AlignmentAgent or its components expect them
        'fail_safe_action_space': ['safe_action1', 'safe_action2'] # For _reduce_agent_privileges
    }

    # --- Mocking get_shared_slailm ---
    class MockSLAILMForAgentTests:
        def __init__(self, *args, **kwargs):
            logger.info("MockSLAILMForAgentTests initialized for AlignmentAgent test")
        def process_input(self, prompt, text, **kwargs):
            logger.info(f"MockSLAILMForAgentTests.process_input called with prompt: {prompt}, text: {text}")
            # Return structure potentially used by ValueEmbeddingModel if it's active
            return {"tokens": ["mock_token"] * 10, "embedding": torch.randn(768)} 

    original_get_shared_slailm_func = None
    try:
        # Ensure models.slai_lm is importable before trying to patch
        import models.slai_lm as slai_lm_module
        if hasattr(slai_lm_module, 'get_shared_slailm'):
            original_get_shared_slailm_func = slai_lm_module.get_shared_slailm
            slai_lm_module.get_shared_slailm = lambda sm, tok, enc, af: MockSLAILMForAgentTests(sm, tok, enc, af)
            logger.info("Patched models.slai_lm.get_shared_slailm for testing.")
        else:
            logger.warning("models.slai_lm does not have get_shared_slailm. SLAILM may not be mocked.")
    except ModuleNotFoundError:
        logger.error("Module 'models.slai_lm' not found. SLAILM cannot be mocked. Tests involving it may fail.")
    except ImportError:
        logger.error("ImportError for 'models.slai_lm'. SLAILM cannot be mocked. Tests involving it may fail.")


    alignment_agent = AlignmentAgent(
        shared_memory=shared_memory_instance,
        agent_factory=agent_factory_instance,
        shared_tokenizer=shared_tokenizer_instance,
        shared_text_encoder=shared_text_encoder_instance,
        sensitive_attrs=sensitive_attrs,
        config=test_config
    )
    
    print("\n=== Test 1: Basic Alignment Check ===")
    sample_data_verify = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'gender': [1, 0, 1, 0, 1],
        'ethnicity': [2, 1, 3, 2, 1],
        'feature': [0.5, 0.7, 0.3, 0.6, 0.2],
        'input': np.random.rand(5), 
        'context': np.random.rand(5) 
    })
    predictions_verify = np.array([0.6, 0.4, 0.8, 0.3, 0.7])
    labels_verify = np.array([1, 0, 1, 0, 1])

    try:
        verification_report = alignment_agent.verify_alignment({
            'inputs': sample_data_verify,
            'predictions': predictions_verify,
            'labels': labels_verify,
            'action_context': {
                'decision_engine': {'is_active': True}, 
                'action_parameters': {'kinetic_energy': 10}
            }
        })
        print("Verification Result:")
        print(f"Approved: {verification_report.get('approved')}")
        risk_bd = verification_report.get('risk_breakdown', 'N/A')
        if isinstance(risk_bd, float):
             print(f"Risk Score: {risk_bd:.4f}")
        else:
             print(f"Risk Breakdown: {risk_bd}")
        print("Corrections:", verification_report.get('corrections'))
    except Exception as e:
        print(f"Test 1 (Basic Alignment Check) failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Test 2: Error Handling Test ===")
    try:
        print("Testing known failure pattern (string input)...")
        # This will call AlignmentAgent.perform_task due to override
        result_str = alignment_agent.execute("this task should fail") 
        print(f"Result for 'fail' string: {result_str}")

        print("Testing known failure pattern (dict input)...")
        result_dict = alignment_agent.execute({"input": "data", "context": "info", "text": "fail this one"})
        print(f"Result for 'fail' dict: {result_dict}")

        print("Testing successful execution (dict input)...")
        result_ok = alignment_agent.execute({"input": "good data", "context": "normal context"})
        print(f"Result for successful dict: {result_ok}")

    except Exception as e:
        print(f"Error caught during Test 2: {str(e)}")
        print("Shared memory errors:", shared_memory_instance.get(f"errors:{alignment_agent.name}"))


    print("\n=== Test 3: Full Alignment Pipeline ===")
    # Data for align method - ensure it has sensitive attributes and other needed columns
    align_test_data = pd.DataFrame({
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice([0,1], 100), # Added gender
        'ethnicity': np.random.choice([1,2,3], 100), # Added ethnicity
        'income': np.random.normal(50000, 15000, 100),
        # Add other columns if model_predict_func (AlignmentAgent.perform_task) expects them
        # when called by AlignmentMonitor
        'input': np.random.rand(100), 
        'context': np.random.rand(100) 
    })
    align_predictions = torch.randn(100)
    align_labels = torch.randint(0, 2, (100,))
    try:
        alignment_report_pipeline = alignment_agent.align(
            data=align_test_data,
            predictions=align_predictions,
            labels=align_labels
        )
        print("Alignment Report (Pipeline):")
        total_risk_val = alignment_report_pipeline.get('risk_assessment', {}).get('total_risk', float('nan'))
        print(f"Total Risk: {total_risk_val:.4f}")
        print("Applied Correction:", alignment_report_pipeline.get('applied_correction'))
    except Exception as e:
        print(f"Test 3 (Full Alignment Pipeline) failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Restore original get_shared_slailm if it was patched
    if original_get_shared_slailm_func and 'models.slai_lm' in sys.modules:
        sys.modules['models.slai_lm'].get_shared_slailm = original_get_shared_slailm_func
        logger.info("Restored models.slai_lm.get_shared_slailm.")

    print("\nAll tests completed (up to the point of fixes)!")
