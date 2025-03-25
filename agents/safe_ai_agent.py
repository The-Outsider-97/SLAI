import logging
import random
import os
import sys
import torch
import numpy as np

class SafeAI_Agent:
    """
    Safety-aware agent that monitors and adjusts other agents' behavior.
    It includes a basic learning model that can improve safety assessments over time.
    """

    def __init__(self, shared_memory=None, risk_threshold=0.2):
        self.name = "SafeAI_Agent"
        self.shared_memory = shared_memory
        self.risk_threshold = risk_threshold
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Risk model: (task_type, risk_score) pairs
        self.training_data = []  # stores past risk data for training
        self.risk_table = {}     # learned safety thresholds per task type

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

    def execute(self, task_data):
        """
        Evaluate risk and propose adjustments.
        """
        policy_score = task_data.get("policy_risk_score", None)
        task_type = task_data.get("task_type", "general")

        if policy_score is None:
            return {
                "status": "failed",
                "error": "Missing 'policy_risk_score' in task_data"
            }

        safe = self.assess_risk(policy_score, task_type)
        correction = self.suggest_correction(policy_score, task_type)

        # Update shared memory with results
        if self.shared_memory:
            self.shared_memory.set("last_policy_risk", policy_score)
            self.shared_memory.set("safe_ai_recommendation", correction or "no_action")

        # Store training data
        self.training_data.append((task_type, policy_score))

        result = {
            "status": "assessed",
            "agent": self.name,
            "risk_score": policy_score,
            "is_safe": safe,
            "recommendation": correction
        }

        self.logger.info(f"[SafeAI Agent] Executed risk assessment: {result}")
        return result

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
