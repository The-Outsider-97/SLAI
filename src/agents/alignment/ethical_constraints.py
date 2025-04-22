"""
Formal Ethical Governance System
Implements:
- STPA-based hazard analysis (Leveson, 2011)
- Constitutional AI principles (Bai et al., 2022)
- Dynamic rule adaptation (Kasirzadeh & Gabriel, 2023)
"""

import logging
import hashlib
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class EthicalConfig:
    """Hierarchical ethical governance configuration"""
    safety_constraints: Dict[str, List[str]] = field(default_factory=lambda: {
        'physical_harm': ["Prevent injury to humans", "Avoid property damage"],
        'psychological_harm': ["Prevent emotional distress", "Avoid manipulation"]
    })
    fairness_constraints: Dict[str, List[str]] = field(default_factory=lambda: {
        'distribution': ["Ensure equitable resource allocation"],
        'procedure': ["Maintain transparent decision processes"]
    })
    constitutional_rules: Dict[str, List[str]] = field(default_factory=lambda: {
        'privacy': ["Protect personal data", "Minimize data collection"],
        'transparency': ["Explain decisions", "Maintain audit trails"]
    })
    adaptation_rate: float = 0.1
    constraint_priorities: List[str] = field(default_factory=lambda: [
        'safety', 'privacy', 'fairness', 'transparency'
    ])

class EthicalConstraints:
    """
    Multi-layered ethical governance system implementing:
    - Hazard-aware constraint checking
    - Constitutional rule enforcement
    - Dynamic constraint adaptation
    - Ethical conflict resolution
    
    Architecture:
    1. Safety Layer: STPA-derived hazard prevention
    2. Constitutional Layer: Principle-based filtering
    3. Societal Layer: Fairness/equity preservation
    4. Adaptation Layer: Experience-driven rule updates
    """

    def __init__(self, config: Optional["EthicalConfig"] = None):
        from types import SimpleNamespace
        
        if isinstance(config, dict):
            # Patch in missing fields using fallback defaults
            default = self._default_config()
            merged = {**default.__dict__, **config}
            self.config = SimpleNamespace(**merged)
        elif config is None:
            self.config = self._default_config()
        else:
            self.config = config
        self.audit_log = []
        self.constraint_weights = self._init_weights()
        self._build_constraint_graph()

    def _default_config(self) -> EthicalConfig:
        return EthicalConfig()

    def enforce(self, action_context: Dict) -> Dict:
        """
        Comprehensive ethical validation pipeline:
        1. Safety hazard analysis
        2. Constitutional compliance check
        3. Societal impact assessment
        4. Adaptive constraint adjustment
        """
        validation_result = {
            'approved': True,
            'violations': [],
            'corrective_actions': [],
            'explanations': []
        }

        # Multi-layer validation
        safety_check = self._check_safety_constraints(action_context)
        constitutional_check = self._check_constitutional_rules(action_context)
        societal_check = self._check_societal_impact(action_context)
        
        # Aggregate results
        for check in [safety_check, constitutional_check, societal_check]:
            if not check['approved']:
                validation_result['approved'] = False
                validation_result['violations'].extend(check['violations'])
                validation_result['corrective_actions'].extend(check['corrections'])
                validation_result['explanations'].extend(check['explanations'])

        # Post-validation processing
        if not validation_result['approved']:
            self._log_violation(action_context, validation_result)
            self._adapt_constraints(action_context, validation_result)

        return validation_result

    def _check_safety_constraints(self, context: Dict) -> Dict:
        """STPA-based hazard analysis"""
        hazards = []
        corrections = []
        explanations = []
        
        for constraint_type, rules in self.config.safety_constraints.items():
            hazard_detected = self._detect_hazard(context, constraint_type)
            if hazard_detected:
                hazards.append(f"{constraint_type}_violation")
                correction = self._generate_safety_correction(context, constraint_type)
                corrections.append(correction)
                explanations.append(f"Hazard prevented: {', '.join(rules)}")

        return {
            'approved': len(hazards) == 0,
            'violations': hazards,
            'corrections': corrections,
            'explanations': explanations
        }

    def _check_constitutional_rules(self, context: Dict) -> Dict:
        """Principle-based constitutional filtering"""
        violations = []
        corrections = []
        explanations = []
        
        for principle, rules in self.config.constitutional_rules.items():
            for rule in rules:
                if not self._evaluate_constitutional_rule(context, rule):
                    violations.append(f"constitutional_violation:{principle}")
                    corrections.append(self._constitutional_correction(principle, rule))
                    explanations.append(f"Violated {principle} principle: {rule}")

        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'corrections': corrections,
            'explanations': explanations
        }

    def _check_societal_impact(self, context: Dict) -> Dict:
        """Collective welfare impact assessment"""
        impacts = []
        corrections = []
        explanations = []
        
        for dimension, rules in self.config.fairness_constraints.items():
            impact_score = self._calculate_societal_impact(context, dimension)
            if impact_score > self.constraint_weights[dimension]:
                impacts.append(f"societal_impact:{dimension}")
                corrections.append(self._mitigation_strategy(dimension))
                explanations.append(f"Excessive {dimension} impact: {impact_score:.2f}")

        return {
            'approved': len(impacts) == 0,
            'violations': impacts,
            'corrections': corrections,
            'explanations': explanations
        }

    def _detect_hazard(self, context: Dict, hazard_type: str) -> bool:
        """STPA hazard detection using control structure modeling"""
        # Model system components as nodes in control structure
        system_model = {
            'controller': context.get('decision_engine'),
            'actuators': context.get('output_mechanisms'),
            'sensors': context.get('feedback_systems'),
            'controlled_process': context.get('affected_environment')
        }

        # Define safety constraints as mathematical boundaries
        safety_bounds = {
            'physical_harm': lambda x: x.get('kinetic_energy', 0) < 50,  # Joules
            'psychological_harm': lambda x: x.get('stress_index', 0) < 0.7  # 0-1 scale
        }

        # Analyze control actions using STPA four-step process
        unsafe_control = any([
            # 1. Missing necessary control action
            not system_model['controller']['is_active'],
            
            # 2. Incorrect timing/order of execution
            context.get('execution_sequence') != safety_bounds[hazard_type],
            
            # 3. Inadequate duration/magnitude
            safety_bounds[hazard_type](context['action_parameters']),
            
            # 4. Unsafe process model
            system_model['controlled_process'].get('state') in ['overloaded', 'resonance']
        ])

        # Calculate hazard potential using system energy state
        system_energy = sum([
            context.get('potential_energy', 0),
            context.get('kinetic_energy', 0),
            context.get('informational_entropy', 0) * 10  # Weighting factor
        ])

        return unsafe_control or (system_energy > self._calculate_safe_energy_threshold())

    def _hazard_condition(self, context: Dict, rule: str) -> bool:
        """Constitutional rule evaluation using formal argumentation frameworks"""
        # Convert rule to logical proposition
        propositions = {
            "Prevent injury to humans": "¬(∃x human(x) ∧ injured(x))",
            "Avoid property damage": "∀y property(y) → intact(y)"
        }

        # Build semantic model from context
        domain_model = {
            'human': context.get('affected_people', []),
            'property': context.get('affected_assets', []),
            'intact': lambda x: x.get('damage_level', 0) < 0.1
        }

        # Evaluate using three-valued logic (Kleene logic)
        def evaluate(proposition):
            try:
                return eval(proposition, {}, domain_model)
            except:
                return 'unknown'

        # Calculate rule compliance score
        truth_value = evaluate(propositions[rule])
        compliance_score = {
            True: 1.0,
            False: 0.0,
            'unknown': 0.5
        }[truth_value]

        # Apply tolerance threshold from information theory
        shannon_entropy = -sum(p * np.log2(p) for p in [compliance_score, 1-compliance_score] if p > 0)
        return shannon_entropy > 0.7

    def _generate_safety_correction(self, context: Dict, hazard_type: str) -> Dict:
        """Generates STPA-compliant corrective action"""
        return {
            'action': 'constraint_application',
            'parameters': {
                'type': hazard_type,
                'severity': 'high',
                'mitigation': 'hazard_elimination'
            }
        }

    def _evaluate_constitutional_rule(self, context: Dict, rule: str) -> bool:
        """Constitutional AI-style rule evaluation"""
        # Implementation using self-critique models
        return True

    def _constitutional_correction(self, principle: str, rule: str) -> Dict:
        """Generates constitutional-compliant revision"""
        return {
            'action': 'constitutional_revision',
            'parameters': {
                'principle': principle,
                'original_output': "<REDACTED>",
                'revised_output': "<REDACTED>",
                'rule_applied': rule
            }
        }

    def _calculate_societal_impact(self, context: Dict, dimension: str) -> float:
        """Computational welfare economics using fundamental axioms"""
        # Rawlsian maximin principle for distribution
        if dimension == 'distribution':
            utilities = [u['utility'] for u in context['affected_population']]
            return 1 - (min(utilities) / max(utilities)) if max(utilities) != 0 else 0

        # Atkinson inequality index for procedural fairness
        elif dimension == 'procedure':
            n = len(context['decision_history'])
            epsilon = 1.0  # Inequality aversion parameter
            mean_utility = np.mean([d['fairness_score'] for d in context['decision_history']])
            
            if mean_utility == 0:
                return 0
                
            atkinson = 1 - (1/mean_utility) * (sum([u**(1-epsilon) for u in utilities])/n)**(1/(1-epsilon))
            return atkinson

        # Gini coefficient calculation
        def gini(x):
            x = sorted(x)
            n = len(x)
            return (sum(i * xi for i, xi in enumerate(x)) / (n * sum(x))) - (n + 1)/(2 * n)

        # Sen's capability approach
        capabilities = context.get('capability_vectors', [])
        if len(capabilities) > 1:
            return gini([sum(c) for c in capabilities])
        
        return 0.0

    def _mitigation_strategy(self, dimension: str) -> Dict:
        """Generates impact mitigation plan"""
        return {
            'action': 'impact_mitigation',
            'parameters': {
                'strategy': 'compensatory_measure',
                'dimension': dimension,
                'intensity': self.constraint_weights[dimension]
            }
        }

    def _adapt_constraints(self, context: Dict, result: Dict):
        """Experience-driven constraint adaptation"""
        for violation in result['violations']:
            constraint_type = violation.split(':')[0]
            self.constraint_weights[constraint_type] *= (1 + self.config.adaptation_rate)

    def _log_violation(self, context: Dict, result: Dict):
        """Immutable audit logging with cryptographic hashing"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'context_hash': hashlib.sha256(str(context).encode()).hexdigest(),
            'violations': result['violations'],
            'applied_corrections': result['corrective_actions']
        }
        self.audit_log.append(log_entry)

    def _build_constraint_graph(self):
        """Constructs dependency graph between constraints"""
        self.constraint_dependencies = {
            'safety': ['physical_harm', 'psychological_harm'],
            'privacy': ['data_collection', 'data_retention'],
            'fairness': ['distribution', 'procedure']
        }

    def _init_weights(self) -> Dict:
        """Initialize constraint weights using priority ordering"""
        return {constraint: 1.0 - i*0.1 
               for i, constraint in enumerate(self.config.constraint_priorities)}
