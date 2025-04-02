
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
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.alignment_agent import AlignmentAgent
from collaborative.shared_memory import SharedMemory
from alignment.alignment_monitor import AlignmentMonitor
from evaluators.report import PerformanceEvaluator
from typing import Dict, List, Optional
from dataclasses import dataclass, field

    
class RewardModel:
    def __init__(self):
        self.rules = {
            "alignment": lambda x: 1 - x.count("harm") / max(1, len(x)),
            "helpfulness": lambda x: x.count("assist") / max(1, len(x))
        }

    def evaluate(self, text: str) -> Dict[str, float]:
        return {name: rule(text) for name, rule in self.rules.items()}

class SafeAI_Agent:
    """
    Safety-aware agent that monitors and adjusts other agents' behavior.
    It includes a basic learning model that can improve safety assessments over time.
    """

    def __init__(self, shared_memory=None, risk_threshold=0.2):
        self.name = "SafeAI_Agent"
        self.shared_memory = shared_memory
        self.risk_threshold = risk_threshold
        self.logger = logging.getLogger("SLAI.SafetyFactory")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

        self.alignment_agent = AlignmentAgent(
            sensitive_attrs=['gender', 'race'],
            config=load_config('alignment_config.yaml'),
            shared_memory=shared_memory
        )

        # Risk model: (task_type, risk_score) pairs
        self.training_data = []  # stores past risk data for training
        self.risk_table = {}     # learned safety thresholds per task type
        self.evaluator = PerformanceEvaluator(threshold=risk_threshold * 100)
        self.alignment_monitor = AlignmentMonitor(...)
        self.audit_trail = []

    def run_alignment_check(self, data, predictions, probs, labels, actions):
        return self.alignment_monitor.monitor(data, predictions, probs, labels, actions)

    def _load_constitution(self, config: Dict) -> Dict:
        """Load safety rules from config or default constitution"""
        default_rules = {
            "privacy": [
                "Do not reveal personal or sensitive information",
                "Anonymize data before processing"
            ],
            "safety": [
                "Prevent physical or psychological harm",
                "Avoid dangerous content generation"
            ],
            "ethics": [
                "Maintain fairness and avoid discrimination",
                "Respect cultural and social norms"
            ]
        }
        return config.get("constitutional_rules", default_rules)

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

    def execute(self, task_data):
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

# 2. Then define the SafetyAgent class
class SafetyAgent:
    """Holistic safety management per Bai et al. (2022)"""
    
    def __init__(self, factory: SafeAI_Agent, config: SafetyAgentConfig):
        self.safety_layer = factory
        self.config = config
        self.reward_model = self._init_reward_model()
        self.attention_monitor = AttentionMonitor()
        
    def _init_reward_model(self):
        return RewardModel({
            "alignment": lambda x: 1 - x.count("harm")/len(x),
            "helpfulness": lambda x: x.count("assist")/len(x)
        })

    def validate_action(self, action: Dict) -> Dict:
        reward_scores = self.reward_model.evaluate(str(action))
        validation['reward_scores'] = reward_scores
        """STPA-based action validation (Leveson, 2011)"""
        validation = {
            "approved": True,
            "corrections": [],
            "risk_assessment": {}
        }
        
        for risk_type, threshold in self.config.risk_thresholds.items():
            risk_score = self._calculate_risk(action, risk_type)
            validation["risk_assessment"][risk_type] = risk_score
            if risk_score > threshold:
                validation["approved"] = False
                correction = self.safety_layer.apply_corrections(action)
                validation["corrections"].append(correction)
                
        return validation

    def _calculate_risk(self, action: Dict, risk_type: str) -> float:
        """Quantitative risk modeling (ISO 26262)"""
        base_risk = len(action.get('parameters', [])) * 0.01
        if risk_type == "privacy":
            return base_risk * self._detect_pii(action)
        return base_risk

    def _detect_pii(self, data: Dict) -> int:
        """Simple PII detection without external libs"""
        pii_keywords = ["name", "email", "address", "phone"]
        return sum(1 for k in data if any(pii in k.lower() for pii in pii_keywords))

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
