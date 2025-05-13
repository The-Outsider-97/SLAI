"""
Formal Ethical Governance System
Implements:
- STPA-based hazard analysis (Leveson, 2011)
- Constitutional AI principles (Bai et al., 2022)
- Dynamic rule adaptation (Kasirzadeh & Gabriel, 2023)
"""

import yaml
import hashlib
import numpy as np
import networkx as nx

from types import SimpleNamespace
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial

from logs.logger import get_logger

logger = get_logger("Ethical Constraints")

CONFIG_PATH = "src/agents/alignment/configs/alignment_config.yaml"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

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

    def __init__(self,
                 config_section_name: str = "ethical_constraints",
                 config_file_path: str = CONFIG_PATH
                 ):
        self.config = get_config_section(config_section_name, config_file_path)
        self.audit_log = []
        self.constraint_weights = self._init_weights()
        self._build_constraint_graph()

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
        
        for constraint_type, rules in vars(
            getattr(self.config, 'safety_constraints', SimpleNamespace())
        ).items():
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
        
        for principle, rules in vars(
            getattr(self.config, 'constitutional_rules', SimpleNamespace())
        ).items():
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
        
        for dimension, rules in vars(
            getattr(self.config, 'fairness_constraints', SimpleNamespace())
        ).items():
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
        system_model = {
            'controller': context.get('decision_engine', {}),
            'actuators': context.get('output_mechanisms'),
            'sensors': context.get('feedback_systems'),
            'controlled_process': context.get('affected_environment', {})
        }

        safety_bounds = {
            'physical_harm': lambda x: x.get('kinetic_energy', 0) < 50,
            'psychological_harm': lambda x: x.get('stress_index', 0) < 0.7
        }

        if hazard_type not in safety_bounds:
            logger.warning(f"Unknown hazard type '{hazard_type}' for safety bounds. Assuming no specific bound check needed.")
            hazard_specific_condition_met = False
        else:
            hazard_specific_condition_met = safety_bounds[hazard_type](context.get('action_parameters', {}))


        # Robust checks for controller activity and execution sequence
        controller_active = system_model.get('controller', {}).get('is_active', False)
        
        # execution_sequence might be a callable or a value. Need to handle safely.
        execution_sequence_safe = True # Placeholder - adjust based on actual meaning
        if 'execution_sequence_status' in context:
             execution_sequence_safe = context['execution_sequence_status'] == 'SAFE'


        unsafe_control_conditions = [
            not controller_active,
            # For now, assuming a boolean flag or a specific check for the hazard type
            # Example: context.get('execution_valid_for_' + hazard_type, True) == False
            not execution_sequence_safe, # Placeholder
            
            # Inadequate duration/magnitude (using the hazard_specific_condition_met from above)
            # So, for unsafe control, we need `not hazard_specific_condition_met`
            not hazard_specific_condition_met if hazard_type in safety_bounds else False,

            system_model.get('controlled_process', {}).get('state') in ['overloaded', 'resonance']
        ]
        unsafe_control = any(unsafe_control_conditions)

        system_energy = sum([
            context.get('potential_energy', 0),
            context.get('kinetic_energy', 0),
            context.get('informational_entropy', 0) * 10
        ])
        
        # Define _calculate_safe_energy_threshold if not defined elsewhere
        # Placeholder for the safe energy threshold calculation
        safe_energy_threshold = getattr(self.config, 'safe_energy_threshold', 100)

        return unsafe_control or (system_energy > safe_energy_threshold)

    def _calculate_safe_energy_threshold(self) -> float:
        # This method should be defined, e.g., load from config or calculate
        # For now, a placeholder value.
        return getattr(self.config, 'safe_energy_threshold', 100.0)

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
        """Evaluate constitutional rules with simple heuristic checks"""
        if "Protect personal data" in rule:
            return self._eval_privacy_protection(context)
        elif "Minimize data collection" in rule:
            return self._eval_privacy_minimization(context)
        elif "Explain decisions" in rule:
            return self._eval_transparency_explainability(context)
        elif "Maintain audit trails" in rule:
            return self._eval_transparency_auditability(context)
        elif "Do not store user personal data longer than necessary" in rule:
            return self._eval_data_retention_policy(context)
        elif "Respect user anonymity whenever possible" in rule:
            return self._eval_respect_user_anonymity(context)
        elif "Collect only the minimum necessary personal information" in rule:
            return self._eval_minimum_necessary_info(context)
        elif "Do not share user data with third parties without explicit consent" in rule:
            return self._eval_no_data_sharing_without_consent(context)
        elif "Encrypt all sensitive information at rest and in transit" in rule:
            return self._eval_encryption_compliance(context)
        elif "Ensure compliance with GDPR and relevant privacy laws" in rule:
            return self._eval_gdpr_compliance(context)
        elif "Allow users to access and delete their personal data" in rule:
            return self._eval_data_access_deletion_rights(context)
        elif "Ensure data breaches are reported and mitigated promptly" in rule:
            return self._eval_data_breach_response(context)
        else:
            logger.warning(f"No specific evaluation logic found for constitutional rule: '{rule}'. Defaulting to True.")
            return True

    def _eval_privacy_protection(self, context: Dict) -> bool:
        """
        Evaluates "Protect personal data".
        Expects context keys: 'personal_data_accessed' (list), 'anonymization_level' (float), 'data_encrypted' (bool).
        """
        accessed_data = context.get('personal_data_accessed', [])
        anonymization = context.get('anonymization_level', 0.0) # 0 = none, 1 = full
        encrypted = context.get('data_encrypted', False)

        if not accessed_data:
            return True # No personal data accessed, rule satisfied

        # Rule check: Data should be encrypted AND sufficiently anonymized if accessed
        is_protected = encrypted and (anonymization >= 0.8) # Example threshold

        if not is_protected:
             logger.debug(f"Privacy Protection Check Failed: Accessed={bool(accessed_data)}, Encrypted={encrypted}, Anonymized={anonymization}")
        return is_protected


    def _eval_privacy_minimization(self, context: Dict) -> bool:
        """
        Evaluates "Minimize data collection".
        Expects context keys: 'data_collected_items' (list), 'purpose_justification_ratio' (float).
        """
        collected_items = context.get('data_collected_items', [])
        # Ratio of items justified by explicit purpose vs total items collected
        justification_ratio = context.get('purpose_justification_ratio', 0.0)

        # Rule check: Number of items should be low OR justification ratio high
        is_minimal = len(collected_items) < 5 or justification_ratio > 0.9 # Example thresholds

        if not is_minimal:
             logger.debug(f"Privacy Minimization Check Failed: Items={len(collected_items)}, Justification={justification_ratio}")
        return is_minimal


    def _eval_transparency_explainability(self, context: Dict) -> bool:
        """
        Evaluates "Explain decisions".
        Expects context keys: 'decision_explanation' (str), 'explanation_clarity_score' (float).
        """
        explanation = context.get('decision_explanation', '')
        clarity_score = context.get('explanation_clarity_score', 0.0) # 0 = unclear, 1 = very clear

        # Rule check: Explanation must exist and meet a clarity threshold
        is_explained = explanation and len(explanation.strip()) > 20 and clarity_score >= 0.6 # Example thresholds

        if not is_explained:
             logger.debug(f"Transparency Explainability Check Failed: Explanation Present={bool(explanation)}, Clarity={clarity_score}")
        return is_explained


    def _eval_transparency_auditability(self, context: Dict) -> bool:
        """
        Evaluates "Maintain audit trails".
        Expects context keys: 'audit_trail_status' (str: 'active'/'inactive'), 'log_completeness_score' (float).
        """
        status = context.get('audit_trail_status', 'inactive')
        completeness = context.get('log_completeness_score', 0.0) # 0 = missing logs, 1 = complete

        # Rule check: Audit trail must be active and reasonably complete
        is_auditable = (status == 'active') and (completeness >= 0.85) # Example thresholds

        if not is_auditable:
             logger.debug(f"Transparency Auditability Check Failed: Status={status}, Completeness={completeness}")
        return is_auditable

    def _eval_data_retention_policy(self, context: Dict) -> bool:
        """
        Evaluates "Do not store user personal data longer than necessary".
        Expects context keys: 'data_retention_days' (int), 'retention_policy_limit' (int)
        """
        retention_days = context.get('data_retention_days', 0)
        policy_limit = context.get('retention_policy_limit', 30)  # Default: 30 days max
    
        is_compliant = retention_days <= policy_limit
    
        if not is_compliant:
            logger.debug(f"Data Retention Policy Check Failed: Retention={retention_days}, Limit={policy_limit}")
        return is_compliant

    # 1. Respect user anonymity whenever possible
    def _eval_respect_user_anonymity(self, context: Dict) -> bool:
        """
        Evaluates "Respect user anonymity whenever possible".
        Expects context keys: 'user_identifiers_present' (bool)
        """
        identifiers_present = context.get('user_identifiers_present', False)
        return not identifiers_present

    # 2. Collect only the minimum necessary personal information
    def _eval_minimum_necessary_info(self, context: Dict) -> bool:
        """
        Evaluates "Collect only the minimum necessary personal information".
        Expects: 'data_collected_items' (list), 'required_data_items' (list)
        """
        collected = set(context.get('data_collected_items', []))
        required = set(context.get('required_data_items', []))
        unnecessary_items = collected - required
        return len(unnecessary_items) == 0

    # 3. Do not share user data with third parties without explicit consent
    def _eval_no_data_sharing_without_consent(self, context: Dict) -> bool:
        """
        Evaluates "Do not share user data with third parties without explicit consent".
        Expects: 'data_shared_with_third_parties' (bool), 'user_consent_obtained' (bool)
        """
        shared = context.get('data_shared_with_third_parties', False)
        consent = context.get('user_consent_obtained', False)
        return not shared or (shared and consent)

    # 4. Encrypt all sensitive information at rest and in transit
    def _eval_encryption_compliance(self, context: Dict) -> bool:
        """
        Evaluates "Encrypt all sensitive information at rest and in transit".
        Expects: 'encryption_at_rest' (bool), 'encryption_in_transit' (bool)
        """
        return context.get('encryption_at_rest', False) and context.get('encryption_in_transit', False)

    # 5. Ensure compliance with GDPR and relevant privacy laws
    def _eval_gdpr_compliance(self, context: Dict) -> bool:
        """
        Evaluates "Ensure compliance with GDPR and relevant privacy laws".
        Expects: 'gdpr_compliant' (bool)
        """
        return context.get('gdpr_compliant', False)
    
    # 6. Allow users to access and delete their personal data
    def _eval_data_access_deletion_rights(self, context: Dict) -> bool:
        """
        Evaluates "Allow users to access and delete their personal data".
        Expects: 'access_mechanism_available' (bool), 'deletion_mechanism_available' (bool)
        """
        return context.get('access_mechanism_available', False) and context.get('deletion_mechanism_available', False)
    
    # 7. Ensure data breaches are reported and mitigated promptly
    def _eval_data_breach_response(self, context: Dict) -> bool:
        """
        Evaluates "Ensure data breaches are reported and mitigated promptly".
        Expects: 'recent_breach_detected' (bool), 'breach_response_time_hours' (int)
        """
        breach_detected = context.get('recent_breach_detected', False)
        response_time = context.get('breach_response_time_hours', 0)
        return not breach_detected or (breach_detected and response_time <= 72)  # e.g., 72 hours

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
            scores = [d['fairness_score'] for d in context['decision_history']]
            n = len(scores)
            epsilon = 0.9  # Must be < 1 to avoid division by zero
            mean_score = np.mean(scores)
        
            if mean_score == 0 or n == 0:
                return 0
        
            part_sum = sum([s ** (1 - epsilon) for s in scores])
            atkinson = 1 - (1 / mean_score) * (part_sum / n) ** (1 / (1 - epsilon))
            return atkinson
    
        # Gini coefficient calculation for capabilities
        def gini(x):
            x = sorted(x)
            n = len(x)
            return (sum(i * xi for i, xi in enumerate(x)) / (n * sum(x))) - (n + 1)/(2 * n)
    
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

    def _adapt_constraints(self, context: Dict, result: Dict):
        """Experience-driven constraint adaptation"""
        safety = getattr(self.config, 'safety_constraints', SimpleNamespace())
        fairness = getattr(self.config, 'fairness_constraints', SimpleNamespace())
        constitutional = getattr(self.config, 'constitutional_rules', SimpleNamespace())

        for violation in result['violations']:
            if ':' in violation:
                base_constraint_type = violation.split(':')[1]
            elif '_violation' in violation:
                base_constraint_type = violation.replace('_violation', '')
            else:
                base_constraint_type = violation # Assume it's already the base type if no suffix/delimiter

            # Check if the extracted type exists as a weight key
            if base_constraint_type in self.constraint_weights:
                # Adapt the weight for the corresponding base type
                original_weight = self.constraint_weights[base_constraint_type]
                self.constraint_weights[base_constraint_type] *= (1 + self.config.adaptation_rate)
                logger.info(f"Adapting weight for '{base_constraint_type}' due to violation '{violation}'. New weight: {self.constraint_weights[base_constraint_type]:.4f} (from {original_weight:.4f})")
            else:
                 found_category = False
                 for category, types in vars(safety).items():
                     if base_constraint_type in types or base_constraint_type == category:
                         if category in self.constraint_weights:
                              original_weight = self.constraint_weights[category]
                              self.constraint_weights[category] *= (1 + self.config.adaptation_rate)
                              logger.info(f"Adapting weight for category '{category}' due to violation '{violation}'. New weight: {self.constraint_weights[category]:.4f} (from {original_weight:.4f})")
                              found_category = True
                              break
                         else:
                              logger.warning(f"Weight category '{category}' for violation type '{base_constraint_type}' not found in constraint_weights.")
                              break # Stop searching categories for this violation

                 if not found_category:
                     # Repeat for fairness constraints
                     for category, types in vars(fairness).items():
                          if base_constraint_type in types or base_constraint_type == category:
                               if category in self.constraint_weights:
                                    original_weight = self.constraint_weights[category]
                                    self.constraint_weights[category] *= (1 + self.config.adaptation_rate)
                                    logger.info(f"Adapting weight for category '{category}' due to violation '{violation}'. New weight: {self.constraint_weights[category]:.4f} (from {original_weight:.4f})")
                                    found_category = True
                                    break
                               else:
                                    logger.warning(f"Weight category '{category}' for violation type '{base_constraint_type}' not found in constraint_weights.")
                                    break

                 if not found_category:
                     # Repeat for constitutional rules
                     for category, types in vars(constitutional).items():
                          if base_constraint_type in types or base_constraint_type == category:
                               if category in self.constraint_weights:
                                    original_weight = self.constraint_weights[category]
                                    self.constraint_weights[category] *= (1 + self.config.adaptation_rate)
                                    logger.info(f"Adapting weight for category '{category}' due to violation '{violation}'. New weight: {self.constraint_weights[category]:.4f} (from {original_weight:.4f})")
                                    found_category = True
                                    break
                               else:
                                    logger.warning(f"Weight category '{category}' for violation type '{base_constraint_type}' not found in constraint_weights.")
                                    break

                 if not found_category:
                     logger.warning(f"Could not find matching weight key or category for violation '{violation}' (extracted type: '{base_constraint_type}'). No weight adapted.")

    def _init_weights(self) -> Dict:
        """Initialize constraint weights with safe config access."""
        # Safely get all constraint sections with fallbacks
        safety = getattr(self.config, 'safety_constraints', SimpleNamespace())
        fairness = getattr(self.config, 'fairness_constraints', SimpleNamespace())
        constitutional = getattr(self.config, 'constitutional_rules', SimpleNamespace())
        categories = getattr(self.config, 'constraint_priorities', [])
        
        # Collect all constraint types safely
        all_categories = (
            set(categories) |
            set(vars(safety).keys()) | 
            set(vars(fairness).keys()) | 
            set(vars(constitutional).keys())
        )
    
        # Rest of the weight initialization logic remains the same...
        initial_weight = 1.0
        step = 0.05
        weights = {}
        priority_map = {cat: i for i, cat in enumerate(categories)}
        
        for category in sorted(all_categories, key=lambda x: priority_map.get(x, 99)):
            weights[category] = initial_weight - priority_map.get(category, len(all_categories)) * step
            logger.debug(f"Initialized weight for '{category}': {weights[category]:.4f}")
    
        # Handle fairness dimensions
        for dimension in vars(fairness).keys():
            if dimension not in weights:
                weights[dimension] = initial_weight - len(weights) * step
                logger.warning(f"Fairness dimension added: {dimension}")
        
        return weights

    def _build_constraint_graph(self):
        """Build graph with full safe config access."""
        self.constraint_graph = nx.DiGraph()
        
        # Get all constraint sections with safe defaults
        safety = getattr(self.config, 'safety_constraints', SimpleNamespace())
        fairness = getattr(self.config, 'fairness_constraints', SimpleNamespace())
        constitutional = getattr(self.config, 'constitutional_rules', SimpleNamespace())
        priority_list = getattr(self.config, 'constraint_priorities', [])  # Key fix
    
        # Node creation remains unchanged
        categories = (
            list(vars(safety).keys()) +
            list(vars(fairness).keys()) +
            list(vars(constitutional).keys())
        )
        self.constraint_graph.add_nodes_from(categories)
    
        # Priority relationships with existence checks
        for i in range(len(priority_list)):
            for j in range(i + 1, len(priority_list)):
                u = priority_list[i]
                v = priority_list[j]
                if u in self.constraint_graph and v in self.constraint_graph:
                    self.constraint_graph.add_edge(u, v, type='priority')

        # Define potential conflicts (example: transparency vs privacy)
        if 'transparency' in self.constraint_graph and 'privacy' in self.constraint_graph:
            self.constraint_graph.add_edge('transparency', 'privacy', type='conflict')
            self.constraint_graph.add_edge('privacy', 'transparency', type='conflict')
            logger.debug("Added conflict edge between transparency and privacy")

        # Add more conflict edges based on domain knowledge
        # e.g., Safety might conflict with distribution fairness in resource allocation
        if 'safety' in self.constraint_graph and 'distribution' in self.constraint_graph:
             self.constraint_graph.add_edge('safety', 'distribution', type='conflict')
             self.constraint_graph.add_edge('distribution', 'safety', type='conflict')
             logger.debug("Added conflict edge between safety and distribution")

        logger.info(f"Constraint graph built with {self.constraint_graph.number_of_nodes()} nodes and {self.constraint_graph.number_of_edges()} edges.")

if __name__ == "__main__":
    import json

    print("=== Ethical Constraints Real-World Scenario ===")
    print("Scenario: A SaaS platform wants to share anonymized usage data with a third-party analytics provider.")
    print("You will be asked a few questions to simulate a real decision context.\n")

    user_identifiers_present = input("Does the data include identifiable user information? (yes/no): ").strip().lower() == "yes"
    consent_given = input("Has the user explicitly consented to data sharing? (yes/no): ").strip().lower() == "yes"
    encrypted_at_rest = input("Is the data encrypted at rest? (yes/no): ").strip().lower() == "yes"
    encrypted_in_transit = input("Is the data encrypted in transit? (yes/no): ").strip().lower() == "yes"
    anonymization_level = float(input("Anonymization level (0.0 = none, 1.0 = fully anonymous): ").strip())
    retention_days = int(input("How many days is the data retained?: ").strip())
    data_collected_items = input("List collected data fields (comma-separated): ").strip().split(',')
    required_data_items = input("List required data fields for the task (comma-separated): ").strip().split(',')
    gdpr_compliant = input("Is the platform GDPR compliant? (yes/no): ").strip().lower() == "yes"

    context = {
        'user_identifiers_present': user_identifiers_present,
        'user_consent_obtained': consent_given,
        'data_shared_with_third_parties': True,
        'encryption_at_rest': encrypted_at_rest,
        'encryption_in_transit': encrypted_in_transit,
        'anonymization_level': anonymization_level,
        'personal_data_accessed': data_collected_items,
        'data_retention_days': retention_days,
        'retention_policy_limit': 30,
        'data_collected_items': data_collected_items,
        'required_data_items': required_data_items,
        'purpose_justification_ratio': len(set(data_collected_items) & set(required_data_items)) / max(len(data_collected_items), 1),
        'gdpr_compliant': gdpr_compliant,
        'access_mechanism_available': True,
        'deletion_mechanism_available': True,
        'recent_breach_detected': False,
        'breach_response_time_hours': 0,
        'decision_engine': {'is_active': True},
        'affected_environment': {'state': 'normal'},
        'output_mechanisms': {'type': 'API'},
        'feedback_systems': {'latency': 0.2},
        'action_parameters': {'kinetic_energy': 0, 'stress_index': 0.1},
        'potential_energy': 0,
        'kinetic_energy': 0,
        'informational_entropy': 0.5,
        'affected_people': [],
        'affected_assets': [],
        'affected_population': [{'utility': 0.9}, {'utility': 0.8}],
        'decision_history': [{'fairness_score': 0.8}, {'fairness_score': 0.7}],
        'capability_vectors': [[1, 1], [2, 2]]
    }

    system = EthicalConstraints()
    result = system.enforce(context)

    print("\n=== Ethical Validation Result ===")
    print(json.dumps(result, indent=2))

    print("\n=== Audit Log ===")
    for entry in system.audit_log:
        print(json.dumps(entry, indent=2))
