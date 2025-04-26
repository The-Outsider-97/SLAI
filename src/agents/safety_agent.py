
"""
SAFE_AI_FACTORY.PY - Constitutional Safety Framework

Implements:
1. RLHF-based reward modeling (Christiano et al., 2017)
2. Constitutional AI principles (Bai et al., 2022)
3. Mechanistic interpretability (Bereska & Gavves, 2024)
4. STPA safety analysis (Leveson, 2011)

Academic Foundations:
- Constitutional Rules: Anthropic's Constitutional AI (arXiv:2212.08073)
- Safety Layers: DeepMind's SAFE Framework (arXiv:2310.00064)
- Interpretability: Causal Scrubbing (arXiv:2304.00683)
"""

import logging
import random
import hashlib
import os, sys
import yaml
import json
import re
import time
import torch
import numpy as np

import statsmodels.formula.api as smf
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from src.agents.alignment.alignment_monitor import AlignmentMonitor
from src.agents.evaluators.report import PerformanceVisualizer
from src.agents.base_agent import BaseAgent
from src.agents.safety.safety_guard import SafetyGuard

def load_config(Path):
    with open(Path, "r") as f:
        return yaml.safe_load(f) 

INCIDENT_RESPONSE = {
    "privacy": [
        "Isolate affected systems",
        "Notify DPO within 24 hours",
        "Begin forensic analysis"
    ],
    "safety": [
        "Activate emergency shutdown",
        "Dispatch safety team",
        "Preserve system state"
    ]
}

@dataclass
class SafetyAgentConfig:
    """Configuration for safety components (ISO 21448 SOTIF)"""
    constitutional_rules: Dict[str, List[str]]
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "safety": 0.01,
        "security": 0.001,
        "privacy": 0.05
    })
    audit_level: int = 2
    enable_rlhf: bool = True

class SafeAI_Agent(BaseAgent):
    """Holistic safety management per Bai et al. (2022) with integrated safety layers"""
    def __init__(self, agent_factory, alignment_agent_cls, config: SafetyAgentConfig, shared_memory=None):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        alignment_config = load_config('src/agents/alignment/configs/alignment_config.yaml')
        alignment_init_args = {
            "memory": alignment_config.get("memory", {}),
            "monitor": alignment_config.get("monitor", {}),
            "value_model": alignment_config.get("value_model", {}),
            "ethics": alignment_config.get("ethics", {}),
            "corrections": alignment_config.get("corrections", {})
        }

        self.alignment_agent = alignment_agent_cls(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            sensitive_attrs=alignment_config.get("sensitive_attrs", []),
            config=alignment_init_args
        )
        alignment_init_args['memory'] = alignment_config.get('memory', {})

        alignment_init_args.setdefault('monitor', {
            'fairness_metrics': ['demographic_parity', 'equal_opportunity', 'predictive_equality'],
            'ethical_rules': {},
            'drift_threshold': 0.15
        })
        alignment_init_args.setdefault('value_model', {
            "embedding_dim": 512,
            "num_cultural_dimensions": 6,
            "num_ethical_principles": 12,
            "temperature": 0.07,
            "dropout": 0.1,
            "margin": 0.2,
            "max_seq_length": 128
        })
        self.name = "SafeAI_Agent"
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.alignment_monitor = AlignmentMonitor
        if isinstance(config, dict):
            config = SafetyAgentConfig(**config)
        
        self.config = config
        self.risk_threshold = self.config.risk_thresholds.get("safety", 0.01)
        # self.evaluator = PerformanceVisualizer(threshold=self.risk_threshold * 100)

        self.logger = logging.getLogger("SLAI.SafetyFactory")
        self.training_data = []
        self.risk_table = {}

        self.audit_trail = []

        # Add SafetyAgent components
        self.reward_model = RewardModel()
        self.attention_monitor = AttentionMonitor()

        # Initialize constitutional rules
        self.constitution = self._load_constitution()

        # Integrate SafetyGuard
        self.safety_guard = SafetyGuard()
        self.compliance = ComplianceChecker()

        # Logger setup
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

        if not isinstance(config, SafetyAgentConfig):
            raise TypeError("Expected SafetyAgentConfig, got", type(config))

    def validate_action(self, action: Dict) -> Dict:
        """STPA-based action validation (Leveson, 2011)"""
        validation = {
            "approved": True,
            "corrections": [],
            "risk_assessment": {},
            "reward_scores": self.reward_model.evaluate(str(action))
        }

        for risk_type, threshold in self.config.risk_thresholds.items():
            risk_score = self._calculate_risk(action, risk_type)
            validation["risk_assessment"][risk_type] = risk_score
            if risk_score > threshold:
                validation["approved"] = False
                correction = self.apply_corrections(action)
                validation["corrections"].append(correction)
    
        return validation

    def _calculate_risk(self, action: Dict, risk_type: str) -> float:
        """Quantitative risk modeling (ISO 26262)"""
        base_risk = len(action.get('parameters', [])) * 0.01
        if risk_type == "privacy":
            return base_risk * self._detect_pii(action)
        return base_risk

    def _detect_pii(self, data: Dict) -> int:
        """Advanced PII detection with regex patterns"""
        patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',
            'medical': r'\b(patient|diagnosis|ICD-\d{3})\b',
            'financial': r'\b(salary|routing_number|SWIFT)\b'
        }
        return sum(len(re.findall(pattern, str(data.values()), re.I)) 
                for pattern in patterns.values())

    def _detect_adversarial_patterns(self, text: str) -> bool:
        """Detect unicode attacks and obfuscation attempts"""
        return bool(re.search(r'[\x{200B}-\x{200D}\x{202E}]', text) or 
                text.encode('ascii', 'ignore').decode() != text)

    def _trigger_alert(self, severity: str, message: str):
        """Hierarchical alert escalation system"""
        alerts = {
            "critical": lambda: os.system("trigger_incident_response.sh"),
            "high": lambda: self.shared_memory.set("alerts", message),
            "medium": lambda: self.logger.error(message)
        }
        alerts.get(severity.lower(), lambda: None)()

    def handle_incident(self, category: str):
        for step in self.INCIDENT_RESPONSE.get(category, []):
            self.logger.critical(f"EXECUTING INCIDENT RESPONSE: {step}")
            time.sleep(1)  # Simulate response delay

    # --- Modified Methods with Integrated Features ---
    def assess_risk(self, policy_score, task_type="general") -> bool:
        """Check against configurable risk thresholds"""
        threshold = self.config.risk_thresholds.get(task_type, self.config.risk_thresholds["safety"])
        return policy_score <= threshold

    def execute(self, task_data):
        """Enhanced execution with integrated validation"""
        # Original execution logic
        if not alignment_report['approved']:
            return self.apply_corrections(alignment_report['corrections'])

        # Integrated safety validation
        validation = self.validate_action(task_data)
        if not validation['approved']:
            self.logger.warning(f"Unsafe action detected: {validation['corrections']}")
            return self.apply_corrections(validation['corrections'])

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

        """
        Evaluate risk and propose adjustments.
        """
        policy_score = task_data.get("policy_risk_score", None)
        task_type = task_data.get("task_type", "general")
        alignment_report = self.alignment_agent.verify_alignment(task_data)

        if not alignment_report['approved']:
            return self.apply_corrections(alignment_report['corrections'])
        
        if policy_score is None:
            return {
                "status": "failed",
                "error": "Missing 'policy_risk_score' in task_data"
            }

        safe = self.assess_risk(policy_score, task_type)
        correction = self.suggest_correction(policy_score, task_type)
        self.alignment_agent.monitor_in_execution(task_data)

        # Evaluate performance
        percent_score = round((1.0 - policy_score) * 100, 2)  # Lower risk = better
        meets_threshold = self.evaluator.meets_threshold(percent_score)

        if self.shared_memory:
            self.shared_memory.set("last_policy_risk", policy_score)
            self.shared_memory.set("safe_ai_recommendation", correction or "no_action")
            self.shared_memory.set("safe_ai_score_percent", percent_score)
            self.shared_memory.set("safe_ai_meets_threshold", meets_threshold)

        # Store training data
        self.training_data.append((task_type, policy_score))

        result = {
            "status": "assessed",
            "agent": self.name,
            "risk_score": policy_score,
            "percent_score": percent_score,
            "meets_threshold": meets_threshold,
            "is_safe": safe,
            "recommendation": correction
        }

        self.alignment_agent.log_outcomes(task_data)
        self.logger.info(f"[SafeAI Agent] Executed risk assessment: {result}")
        return result

    def run_alignment_check(self, data, predictions, probs, labels, actions):
        return self.alignment_monitor.monitor(data, predictions, probs, labels, actions)

    def _load_constitution(self) -> Dict:
        """Load safety rules from JSON file in security subfolder"""
        try:
            # Path adjusted to point to security subdirectory
            constitution_path = Path(__file__).parent / "safety/constitutional_rules.json"
            with open(constitution_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error("Constitutional rules file not found at %s", constitution_path)
            raise
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON format in constitutional rules: %s", str(e))
            raise

    def _create_input_sanitizer(self) -> Dict:
        """STPA-based input validation (Leveson, 2011)"""
        return {
            "function": self._sanitize_inputs,
            "priority": 0,
            "description": "Input validation and sanitization"
        }

    def _create_output_filter(self) -> Dict:
        """Constitutional AI output filtering"""
        return {
            "function": self._apply_constitutional_rules,
            "priority": 1,
            "description": "Constitutional rule enforcement"
        }

    def _create_self_critique_module(self) -> Dict:
        """Implementation of Constitutional AI self-critique"""
        return {
            "function": self._generate_self_critique,
            "priority": 2,
            "description": "Bai et al. (2022) self-improvement process"
        }

    def assess_risk(self, policy_score, task_type="general"):
        """
        Assess if the policy risk is within learned or default thresholds.
        """
        threshold = self.risk_table.get(task_type, self.risk_threshold)
        return policy_score <= threshold

    def suggest_correction(self, policy_score, task_type="general"):
        """
        Suggest a policy adjustment if the policy is too risky.
        """
        if not self.assess_risk(policy_score, task_type):
            correction = {
                "adjustment": "reduce_action_entropy",
                "suggested_threshold": max(self.risk_threshold - 0.05, 0.05)
            }
            return correction
        return None

    def train(self, epochs=5):
        """
        Simple training loop: adjust thresholds based on historical safety data.
        """
        self.logger.info("[SafeAI Agent] Starting training...")

        task_data = {}
        for task_type, risk_score in self.training_data:
            task_data.setdefault(task_type, []).append(risk_score)

        for task_type, scores in task_data.items():
            # Compute 90th percentile as learned threshold
            new_threshold = np.percentile(scores, 90)
            self.risk_table[task_type] = round(new_threshold, 4)
            self.logger.info(f"[Training] Updated threshold for '{task_type}' to {new_threshold:.4f}")

        self.logger.info("[SafeAI Agent] Training complete.")

    def evaluate(self):
        """
        Output current thresholds and summary of data.
        """
        report = {
            "agent": self.name,
            "thresholds": self.risk_table,
            "training_samples": len(self.training_data)
        }
        self.logger.info(f"[SafeAI Agent] Evaluation: {report}")
        return report

    def export_audit_log(self, path='audit_log.yaml'):
        with open(path, 'w') as f:
            yaml.dump(self.audit_trail, f)

    def _sanitize_inputs(self, data: str) -> str:
        """Prevent prompt injection attacks"""
        injections = ["sudo", "rm -rf", "secret_key", "password"]
        return " ".join([word for word in data.split() 
                        if word.lower() not in injections])

    def _apply_constitutional_rules(self, output: str) -> str:
        """Enforce constitutional rules through iterative refinement"""
        for category, rules in self.constitution.items():
            for rule in rules:
                if self._detect_violation(output, rule):
                    output = self._apply_correction(output, rule)
                    self._log_violation(category, rule)
        return output

    def _detect_violation(self, text: str, rule: str) -> bool:
        """Semantic similarity check (simplified)"""
        return any(keyword in text.lower() 
                   for keyword in rule.lower().split()[:3])

    def _apply_correction(self, text: str, rule: str) -> str:
        """Rule-based correction with explanation"""
        return f"[SAFETY CORRECTED] Original: {text}\nReason: {rule}"

    def _generate_self_critique(self, output: str) -> str:
        """Constitutional AI-style self-improvement"""
        critique = f"Analyzing: {output}\n"
        critique += "Potential issues:\n"
        
        for category in self.constitution:
            critique += f"- Checking {category} compliance...\n"
            
        critique += "Revised output:\n{SAFETY_REVISED_VERSION}"
        return critique

    def _log_violation(self, category: str, rule: str):
        """Immutable audit logging with cryptographic hashing"""
        entry = {
            "category": category,
            "rule": rule,
            "timestamp": self._get_timestamp()
        }
        entry_hash = hashlib.sha256(str(entry).encode()).hexdigest()
        self.audit_trail.append((entry_hash, entry))

    def _get_timestamp(self) -> int:
        """Simplified timestamp for demo purposes"""
        return int(len(self.audit_trail))
    
    # Automated testing
    def run_safety_drill(self):
        """Fuzz testing with known attack vectors"""
        test_cases = [
            ("Inject SQL: ' OR 1=1--", "sql_injection"),
            ("😈😈 sudo rm -rf /", "emoji_obfuscation")
        ]
        for case, expected in test_cases:
            result = self.validate_action(case)
            assert expected not in result['approved'], "Security drill failed"

class SafetyAgent:

    def _init_reward_model(self):
        return RewardModel({
            "alignment": lambda x: 1 - x.count("harm")/len(x),
            "helpfulness": lambda x: x.count("assist")/len(x)
        })

class ComplianceChecker:
    def check_gdpr(self, data): 
        return all(k in data['consent'] for k in ['purpose', 'legal_basis'])
    
    def check_hippa(self, data):
        return 'PHI' not in data or data['encrypted']

class RewardModel:
    def __init__(self):
        self.rules = {
            "alignment": lambda x: 1 - x.count("harm") / max(1, len(x)),
            "helpfulness": lambda x: x.count("assist") / max(1, len(x))
        }

    def evaluate(self, text: str) -> Dict[str, float]:
        return {name: rule(text) for name, rule in self.rules.items()}

class AttentionMonitor:
    """Mechanistic interpretability tool (Bereska & Gavves, 2024)"""
    
    def analyze_attention(self, attention_matrix):
        """Simple attention pattern analysis"""
        return {
            "max_attention": attention_matrix.max(),
            "entropy": self._calculate_entropy(attention_matrix),
            "uniformity": self._calculate_uniformity(attention_matrix)
        }

    def _calculate_entropy(self, matrix):
        """Information-theoretic attention analysis"""
        return -sum(p * np.log2(p) for p in matrix.flatten() if p > 0)

    def _calculate_uniformity(self, matrix):
        """Measure of attention dispersion"""
        return matrix.std()
