import logging
import numpy as np


class SafeAI_Agent:
    """
    A minimal agent for risk-aware policy evaluation and correction.
    Can assess risk levels and suggest policy overrides if thresholds are exceeded.
    """

    def __init__(self, shared_memory=None, risk_threshold=0.2):
        self.name = "SafeAI_Agent"
        self.shared_memory = shared_memory
        self.risk_threshold = risk_threshold
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def assess_risk(self, policy_score):
        """
        Assess if the policy risk is within acceptable bounds.
        """
        return policy_score <= self.risk_threshold

    def suggest_correction(self, policy_score):
        """
        Generate a simple correction recommendation.
        """
        correction = {
            "adjustment": "reduce_action_entropy",
            "new_threshold": max(self.risk_threshold - 0.05, 0.05)
        }
        return correction if policy_score > self.risk_threshold else None

    def execute(self, task_data):
        """
        Entry point for task execution from task router.
        Task should include a 'policy_risk_score'.
        """
        policy_score = task_data.get("policy_risk_score", None)

        if policy_score is None:
            return {
                "status": "failed",
                "error": "Missing 'policy_risk_score' in task_data"
            }

        safe = self.assess_risk(policy_score)
        correction = self.suggest_correction(policy_score)

        # Update shared memory if applicable
        if self.shared_memory:
            self.shared_memory.set("last_policy_risk", policy_score)
            self.shared_memory.set("safe_ai_recommendation", correction or "no_action")

        result = {
            "status": "assessed",
            "agent": self.name,
            "risk_score": policy_score,
            "is_safe": safe,
            "recommendation": correction
        }

        self.logger.info(f"[SafeAI Agent] Executed risk assessment: {result}")
        return result
