
from datetime import datetime
import random
import hashlib
import os, sys
import string
import gym
import re
import time
import torch
import json, yaml
import unicodedata
import numpy as np

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict, dataclass, field, fields

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.safety.utils.secure_stpa import SecureSTPA
from src.agents.safety.reward_model import RewardModel
from src.agents.safety.cyber_safety import CyberSafetyModule
from src.agents.safety.safety_guard import SafetyGuard
from src.agents.safety.compliance_checker import ComplianceChecker
from src.agents.safety.attention_monitor import AttentionMonitor
from src.agents.safety.adaptive_security import AdaptiveSecurity
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Safety Agent")
printer = PrettyPrinter

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

class SafetyAgent(BaseAgent):
    """
    Holistic safety management agent integrating multiple specialized safety modules.
    It assesses inputs, validates actions, monitors compliance, and applies constitutional AI principles.
    """
    def __init__(self, agent_factory, shared_memory, config=None):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config)

        self.name = "Safety_Agent"
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.safety_config = get_config_section('safety_agent')
        
        self.constitutional_rules_path = self.safety_config.get('constitutional_rules_path')
        self.audit_level = self.safety_config.get('audit_level')
        self.collect_feedback = self.safety_config.get('collect_feedback')
        self.enable_learnable_aggregation = self.safety_config.get('enable_learnable_aggregation')
        self.architecture_map = self.safety_config.get('architecture_map', {})
        self.system_models = self.safety_config.get('system_models', {})
        self.known_hazards = self.safety_config.get('known_hazards', [])
        self.global_losses = self.safety_config.get('global_losses', [])
        self.safety_policies = self.safety_config.get('safety_policies', [])
        self.formal_specs = self.safety_config.get('formal_specs', {})
        self.fault_tree_config = self.safety_config.get('fault_tree_config', {})
        self.risk_thresholds = self.safety_config.get('risk_thresholds', {
            'overall_safety', 'cyber_risk', 'compliance_failure_is_blocker'
        })
        self.secure_memory_config = self.safety_config.get('secure_memory',
            field(default_factory=lambda: {"default_ttl_seconds"}
        ))

        # Add SafetyAgent components
        self.reward_model = RewardModel()
        self.attention_monitor = AttentionMonitor()
        self.safety_guard = SafetyGuard()
        self.secure_stpa = SecureSTPA()
        self.compliance_checker = ComplianceChecker()
        self.adaptive_security = AdaptiveSecurity()
        self.cyber_safety = CyberSafetyModule()

        # Initialize constitutional rules
        self.constitution = self._load_constitution()
        self.training_data = []
        self.risk_table = {}
        self.audit_trail = []

        self.learning_factory = self._init_learning_factory()

        logger.info(f"Safety Agent succesfully initialized with: {self.training_data}")

    def _load_constitution(self) -> Dict:
        """Load constitutional rules from the path specified in SafetyAgentConfig."""
        try:
            constitution_path = Path(self.safety_config['constitutional_rules_path'])
            with open(constitution_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            logger.info(f"Constitutional rules loaded successfully from {constitution_path}")
            return rules
        except FileNotFoundError:
            logger.error(f"Constitutional rules file not found at {self.constitutional_rules_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in constitutional rules file {self.constitutional_rules_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading constitutional rules: {e}")
            raise

    def _init_learning_factory(self):
        # Initialize Learning Factory for adaptive risk aggregation
        if self.enable_learnable_aggregation:
            from src.agents.learning.learning_factory import LearningFactory
            self.learning_factory = LearningFactory(
                env=self._create_risk_aggregation_env(),
            )
            self.risk_aggregator = None
        else:
            self.learning_factory = None

    def perform_task(self, data_to_assess: Any, context: Optional[Dict] = None) -> Dict:
        """
        Performs a comprehensive safety and security analysis on the given data.
        This overrides the BaseAgent.perform_task.
        Args:
            data_to_assess: The input data (e.g., text, code, action parameters).
            context: Optional dictionary providing context about the data or originating task.
        Returns:
            A dictionary containing the aggregated safety assessment.
        """
        logger.info(f"[{self.name}] Performing comprehensive safety assessment.")
        if context is None:
            context = {}
        
        assessment_results = {
            "timestamp": time.time(),
            "input_type": str(type(data_to_assess).__name__),
            "overall_recommendation": "proceed_with_caution", # Default
            "is_safe": False,
            "reports": {}
        }

        # 1. Initial Sanitization and Guarding (SafetyGuard)
        sanitized_data_str = str(data_to_assess)
        try:
            # Depth can be configured or passed in context
            sanitization_depth = context.get("sanitization_depth", "full")
            sanitized_data_str = self.safety_guard.sanitize(sanitized_data_str, depth=sanitization_depth)
            assessment_results["reports"]["safety_guard"] = {"status": "success", "output_preview": sanitized_data_str[:100]+"..."}
            if "[SAFETY_BLOCK]" in sanitized_data_str:
                assessment_results["overall_recommendation"] = "block"
                assessment_results["is_safe"] = False
                assessment_results["reports"]["safety_guard"]["details"] = "Content blocked by SafetyGuard policy."
                logger.warning(f"[{self.name}] Input blocked by SafetyGuard: {sanitized_data_str}")
                return assessment_results # Early exit if SafetyGuard blocks
        except Exception as e:
            logger.error(f"[{self.name}] Error during SafetyGuard sanitization: {e}")
            assessment_results["reports"]["safety_guard"] = {"status": "error", "error": str(e)}
            assessment_results["overall_recommendation"] = "block_due_to_error"
            return assessment_results

        # 2. Cyber Safety Analysis (CyberSafetyModule)
        try:
            cyber_context = context.get("cyber_context", "general_input_analysis")
            cyber_report = self.cyber_safety.analyze_input(sanitized_data_str, context=cyber_context)
            assessment_results["reports"]["cyber_safety"] = cyber_report
            if cyber_report.get("risk_score", 0.0) > self.risk_thresholds.get("cyber_risk", 0.5):
                logger.warning(f"[{self.name}] High cyber risk detected: {cyber_report.get('risk_score')}")
                # Decision to block or just warn can be made later based on aggregated score
        except Exception as e:
            logger.error(f"[{self.name}] Error during CyberSafetyModule analysis: {e}")
            assessment_results["reports"]["cyber_safety"] = {"status": "error", "error": str(e)}

        # 3. Adaptive Security (Phishing, Supply Chain - if applicable based on context/data type)
        # This requires more specific data types (email dict, URL string, file path)
        adaptive_security_report = {}
        if isinstance(data_to_assess, dict) and "subject" in data_to_assess and "body" in data_to_assess: # Heuristic for email
            try:
                adaptive_security_report["email_analysis"] = self.adaptive_security.analyze_email(data_to_assess)
            except Exception as e:
                logger.error(f"[{self.name}] Error during email analysis: {e}")
                adaptive_security_report["email_analysis"] = {"status": "error", "error": str(e)}
        elif isinstance(data_to_assess, str) and (data_to_assess.startswith("http:") or data_to_assess.startswith("https:")): # Heuristic for URL
            try:
                adaptive_security_report["url_analysis"] = self.adaptive_security.analyze_url(data_to_assess)
            except Exception as e:
                logger.error(f"[{self.name}] Error during URL analysis: {e}")
                adaptive_security_report["url_analysis"] = {"status": "error", "error": str(e)}
        if adaptive_security_report:
            assessment_results["reports"]["adaptive_security"] = adaptive_security_report

        # 4. Reward Model Evaluation & Matrix Analysis (Ethical/Safety Score)
        try:
            reward_scores = self.reward_model.evaluate(sanitized_data_str, context=context)
            assessment_results["reports"]["reward_model"] = reward_scores
        except Exception as e:
            logger.error(f"[{self.name}] Error during RewardModel evaluation: {e}")
            assessment_results["reports"]["reward_model"] = {"status": "error", "error": str(e)}

        if self.collect_feedback:
            self._request_human_feedback(
                data_to_assess, 
                assessment_results,
                context
            )

        if context and "attention_matrix" in context:
            try:
                attention_report = self.analyze_attention_matrix(
                    context["attention_matrix"], 
                    context
                )
                assessment_results["reports"]["attention_analysis"] = attention_report

                # Apply attention-based adjustments to safety score
                if attention_report.get("anomaly", False):
                    # Penalize safety score based on anomaly severity
                    anomaly_penalty = attention_report["anomaly_score"] * 0.3
                    context["attention_penalty"] = anomaly_penalty
            except Exception as e:
                logger.error(f"[{self.name}] Error during attention analysis: {e}")
                assessment_results["reports"]["attention_analysis"] = {"status": "error", "error": str(e)}

        # 5. Aggregate risk and make final decision (Simplified for now)
        # A more sophisticated aggregation strategy is needed here.
        # Example: Weighted sum or rule-based logic.
        final_safety_score = reward_scores.get("composite", 0.0) if "reward_model" in assessment_results["reports"] and isinstance(assessment_results["reports"]["reward_model"],dict) else 0.0

        # Consider cyber risk
        # cyber_risk = assessment_results["reports"].get("cyber_safety", {}).get("risk_score", 1.0) # Default to high risk if not found
        attention_penalty = context.get("attention_penalty", 0.0)
        final_safety_score -= attention_penalty
        # final_safety_score -= cyber_risk * 0.5 # Penalize for cyber risk

        assessment_results["final_safety_score"] = final_safety_score
        if final_safety_score > self.risk_thresholds.get("overall_safety", 0.75):
            assessment_results["is_safe"] = True
            assessment_results["overall_recommendation"] = "proceed"
        else:
            assessment_results["is_safe"] = False
            assessment_results["overall_recommendation"] = "block_or_review"
            logger.warning(f"[{self.name}] Overall safety score {final_safety_score} below threshold.")

        # Enhanced risk aggregation
        if self.learning_factory and self.risk_aggregator:
            # Prepare input features for risk aggregator
            features = np.array([
                assessment_results["reports"]["safety_guard"].get("block_score", 0),
                assessment_results["reports"]["cyber_safety"].get("risk_score", 0),
                assessment_results["reports"]["reward_model"].get("composite", 0),
                assessment_results["reports"].get("attention_analysis", {}).get("anomaly_score", 0),
                assessment_results["reports"].get("compliance", {}).get("score", 0),
                context.get("risk_context", 0.5) if context else 0.5
            ], dtype=np.float32)

            # Get learnable risk assessment
            final_safety_score = self.risk_aggregator.predict(features)
            assessment_results["is_safe"] = final_safety_score > 0.5
            assessment_results["final_safety_score"] = final_safety_score
            assessment_results["aggregation_method"] = "learnable_model"
        else:
            # Fallback to heuristic aggregation
            assessment_results["final_safety_score"] = final_safety_score
            assessment_results["is_safe"] = final_safety_score > self.risk_thresholds["overall_safety"]
            assessment_results["aggregation_method"] = "heuristic"

        # Optionally apply constitutional rules to output if this agent was generating text
        # For now, this perform_task is about assessment, not generation.

        return assessment_results

    def _request_human_feedback(self, input_data, assessment, context):
        """Simulate human feedback collection"""
        # In real system, this would be a UI interaction
        simulated_feedback = random.uniform(0.7, 1.0)  

        self.collect_human_feedback(
            text=str(input_data),
            model_scores=assessment["reports"]["reward_model"],
            human_rating=simulated_feedback,
            context=context
        )

        # Periodically retrain model
        if random.random() < 0.3:  # 30% chance to train
            self.update_reward_model()

        # Update risk aggregator with human feedback
        if self.learning_factory and self.risk_aggregator:
            features = self._extract_features_from_assessment(assessment)
            self._update_risk_aggregator(features, simulated_feedback)

    def _extract_features_from_assessment(self, assessment):
        """Extracts features from assessment results for learning"""
        return np.array([
            assessment["reports"]["safety_guard"].get("block_score", 0),
            assessment["reports"]["cyber_safety"].get("risk_score", 0),
            assessment["reports"]["reward_model"].get("composite", 0),
            assessment["reports"].get("attention_analysis", {}).get("anomaly_score", 0),
            assessment["reports"].get("compliance", {}).get("score", 0),
            assessment.get("context_risk", 0.5)
        ], dtype=np.float32)

    def _update_risk_aggregator(self, features, human_rating):
        """Updates the risk aggregation model using human feedback"""
        # Convert human rating to reward
        reward = 1.0 if human_rating > 0.7 else -1.0

        # Create training experience
        experience = {
            "state": features,
            "action": [assessment["final_safety_score"]],
            "reward": reward,
            "next_state": features,  # Simplified for this use case
            "done": True
        }

        # Train the model
        self.learning_factory.memory.add(experience)

        # Periodically update agent
        if random.random() < 0.2:
            self.risk_aggregator = self.learning_factory.select_agent(
                {"task_type": "risk_aggregation"}
            )

    def validate_action(self, action_params: Dict, action_context: Optional[Dict] = None) -> Dict:
        """
        Validates a proposed action before execution using STPA-like principles and module checks.
        Args:
            action_params: Dictionary describing the action and its parameters.
            action_context: Optional dictionary providing context for the action.
        Returns:
            A dictionary with 'approved' (bool) and 'details' (findings, scores).
        """
        logger.info(f"[{self.name}] Validating action: {action_params.get('action_name', 'Unknown Action')}")
        if action_context is None:
            action_context = {}

        validation_result = {
            "approved": True, # Default to approved, turn false on issues
            "risk_assessment": {},
            "reward_scores": {},
            "cyber_findings": [],
            "constitutional_concerns": [],
            "details": []
        }
        
        action_description_str = json.dumps(action_params) # For text-based analysis

        # 1. Cyber Safety Pre-Check (analyze_input)
        try:
            cyber_report = self.cyber_safety.analyze_input(action_description_str, context=action_context.get("cyber_context", "action_validation"))
            validation_result["risk_assessment"]["cyber_risk"] = cyber_report.get("risk_score", 0.0)
            validation_result["cyber_findings"] = cyber_report.get("findings", [])
            if cyber_report.get("risk_score", 0.0) > self.risk_thresholds.get("cyber_risk", 0.5):
                validation_result["approved"] = False
                validation_result["details"].append(f"Cyber risk ({cyber_report.get('risk_score')}) exceeds threshold.")
                self._log_audit_event("action_validation_fail", {"reason": "high_cyber_risk", "action": action_params, "report": cyber_report})
        except Exception as e:
            logger.error(f"[{self.name}] Error during CyberSafetyModule.analyze_input for action validation: {e}")
            validation_result["details"].append(f"Error in cyber safety check: {str(e)}")
            validation_result["approved"] = False # Fail safe

        # 2. Reward Model Evaluation (Ethical/Safety alignment)
        try:
            reward_scores = self.reward_model.evaluate(action_description_str, context=action_context)
            validation_result["reward_scores"] = reward_scores
            # Example: check composite score from reward model
            composite_score = reward_scores.get("composite", 0.0)
            # FIX: Access risk_thresholds from dictionary instead of object attribute
            threshold = self.risk_thresholds.get("overall_safety", 0.75)
            if composite_score < threshold: # Assuming lower score is worse
                validation_result["approved"] = False
                validation_result["details"].append(f"Ethical/Safety score ({composite_score}) below threshold.")
                self._log_audit_event("action_validation_fail", {"reason": "low_reward_score", "action": action_params, "scores": reward_scores})
        except Exception as e:
            logger.error(f"[{self.name}] Error during RewardModel.evaluate for action validation: {e}")
            validation_result["details"].append(f"Error in reward model check: {str(e)}")
            validation_result["approved"] = False

        # 3. STPA-inspired checks / Unsafe Control Actions (UCAs)
        # Extract relevant metadata for STPA
        controller_id = "safety_agent"
        action_name = action_params.get("name", "unknown_action")
        execution_state = action_params.get("execution_context", {})
        timestamp = action_params.get("timestamp", datetime.now().isoformat())
        
        # Ensure control structure has been modeled
        if not self.secure_stpa.control_structure:
            self.secure_stpa.model_control_structure(
                structure=self.architecture_map,  # assumed controller/component map
                process_models=self.system_models  # optional
            )
        
        # Dynamically define scope (hazards, losses, constraints)
        if not self.secure_stpa.hazards:
            self.secure_stpa.define_analysis_scope(
                losses=self.global_losses,
                hazards=self.known_hazards,
                constraints=self.safety_policies
            )
        
        # Generate or refresh Unsafe Control Actions (UCAs)
        uca_list = self.secure_stpa.identify_unsafe_control_actions()
        
        # Build contextual decision space around each UCA
        context_tables = self.secure_stpa.build_context_tables(
            formal_spec=self.formal_specs,
            fta_config=self.fault_tree_config
        )
        
        # Identify potential matches between input action and unsafe context entries
        uca_risks = []
        for context_set in context_tables.values():
            for entry in context_set:
                if action_name == entry["control_action"]:
                    match_score = self._assess_contextual_match(entry, execution_state)
                    if match_score > 0.7:
                        uca_risks.append({
                            "uca_id": entry["uca_id"],
                            "hazard_link": entry["guideword"],
                            "score": match_score,
                            "conditions": entry["hazard_conditions"],
                            "variables": entry["process_variables"],
                            "mitigation": self.secure_stpa._generate_mitigation_strategy(entry),
                            "state_constraints": entry["state_constraints"]
                        })
        
        # Log, store, and flag risk if present
        if uca_risks:
            for risk in uca_risks:
                self.memory.log_event("stpa_uca_risk", {
                    "action": action_name,
                    "risk": risk,
                    "agent": controller_id,
                    "timestamp": timestamp
                })
        
            # Add structured alert to response
            alerts.append({
                "type": "unsafe_control_action",
                "severity": "high",
                "action": action_name,
                "evidence": uca_risks
            })

        # 4. Constitutional AI Check (if action involves generating text or making decisions with ethical implications)
        # This is more for outputs, but could be adapted for action parameters if they represent text.
        if "text_parameter" in action_params:
            concerns = self._check_constitutional_violations(action_params["text_parameter"])
            if concerns:
                validation_result["approved"] = False
                validation_result["constitutional_concerns"] = concerns
                validation_result["details"].append(f"Constitutional concerns with action parameters: {concerns}")

        if not validation_result["approved"]:
            logger.warning(f"[{self.name}] Action VALIDATION FAILED. Details: {validation_result['details']}")
            # Potentially apply corrections or suggest alternatives
            corrections = self.apply_corrections(action_params, validation_result)
            validation_result["suggested_corrections"] = corrections
        else:
            logger.info(f"[{self.name}] Action validated successfully.")
            self._log_audit_event("action_validation_success", {"action": action_params})

        # 5. Attention Pattern Analysis (if available)
        if "attention_matrix" in action_context:
            try:
                attention_report = self.analyze_attention_matrix(
                    action_context["attention_matrix"],
                    {"action": action_params, **action_context}
                )
                validation_result["attention_analysis"] = attention_report

                # Apply attention-based validation rules
                if attention_report.get("anomaly", False):
                    validation_result["approved"] = False
                    validation_result["details"].append(
                        f"Attention anomaly detected: {attention_report['security_assessment']['findings']}"
                    )
            except Exception as e:
                logger.error(f"[{self.name}] Attention analysis error: {e}")

        return validation_result

    def _create_risk_aggregation_env(self):
        class RiskAggregationEnv(gym.Env):
            def __init__(self, safety_agent):
                self.safety_agent = safety_agent
                self.observation_space = gym.spaces.Box(
                    low=0, high=1, shape=(6,), dtype=np.float32
                )
                self.action_space = gym.spaces.Discrete(101)  # 101 possible actions (0-100)
                self.current_obs = None
                self.ground_truth = None
                self.step_count = 0
                self.max_steps = 100
    
            def reset(self):
                self.step_count = 0
                self.current_obs, self.ground_truth = self._generate_sample()
                return self.current_obs
    
            def step(self, action):
                self.step_count += 1
                prediction = action / 100.0  # Convert to [0.0, 1.0]
                reward = -abs(prediction - self.ground_truth)
                done = self.step_count >= self.max_steps
                obs, self.ground_truth = self._generate_sample()
                self.current_obs = obs
                return obs, reward, done, {}
    
            def _generate_sample(self):
                """
                Simulate the output of the safety pipeline.
                Replace this with actual data flow if you want to connect real models.
                """
                # Simulated 6D input from submodules
                observation = np.random.rand(6).astype(np.float32)
    
                # Target output (e.g., human-reviewed safety score)
                ground_truth = float(
                    0.4 * observation[0] + 
                    0.2 * observation[1] + 
                    0.1 * observation[2] + 
                    0.1 * observation[3] +
                    0.1 * observation[4] +
                    0.1 * observation[5]
                )
                ground_truth = np.clip(ground_truth, 0, 1)
                return observation, ground_truth
    
        return RiskAggregationEnv(self)

    def _check_for_ucas(self, action_params: Dict[str, Any], action_context: Dict[str, Any]) -> Optional[str]:
        """
        Checks for Unsafe Control Actions (UCAs) based on system roles, privileges, and action criticality.
        
        Returns:
            A string describing the UCA if detected, or None if safe.
        """
        action = action_params.get("action_name", "").lower()
        user_role = action_context.get("user_role", "unknown").lower()
        critical_paths = ["/etc", "/system", "/root", "/secrets", "/config"]
        
        # Example 1: Prevent deletion of critical files by non-admins
        if action == "delete_file":
            path = action_params.get("path", "").lower()
            if any(path.startswith(cp) for cp in critical_paths) and user_role not in ["admin", "superuser"]:
                return f"Non-admin user '{user_role}' attempted to delete critical file: {path}"
    
        # Example 2: Prevent deployment of services with hardcoded secrets
        if action == "deploy_service":
            config = action_params.get("parameters", {}).get("config_data", "")
            if "secret" in config.lower() or "api_key=" in config.lower():
                return "Deployment contains hardcoded secrets in config_data."
    
        # Example 3: Prevent access to privileged operations by guests
        if action in ["shutdown_system", "reboot_node", "reset_database"] and user_role == "guest":
            return f"Guest user attempted privileged operation: {action}"
    
        # Example 4: Ensure multi-user confirmation for dangerous operations
        if action == "drop_database" and not action_context.get("multi_user_confirmation", False):
            return "Dropping database without multi-user confirmation."
    
        # Add additional UCA checks here as needed
        return None

    def apply_corrections(self, action_params: Dict[str, Any], validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggests or applies corrective changes to action_params to resolve safety issues.
        Args:
            action_params: Original parameters of the action.
            validation_result: Result from validate_action including failure reasons.
        Returns:
            List of suggestions or corrected parameter dictionaries.
        """
        corrections = []
    
        # 1. Redact hardcoded secrets if present
        if "parameters" in action_params:
            params = action_params["parameters"]
            if isinstance(params, dict):
                redacted = dict(params)
                for key, value in params.items():
                    if isinstance(value, str) and any(s in value.lower() for s in ["secret", "api_key=", "password"]):
                        redacted[key] = "[REDACTED_FOR_SECURITY]"
                if redacted != params:
                    corrected = dict(action_params)
                    corrected["parameters"] = redacted
                    corrections.append({
                        "message": "Redacted hardcoded secrets from parameters.",
                        "corrected_action": corrected
                    })
    
        # 2. Adjust user role if action requires elevated privileges
        uca_details = [d for d in validation_result.get("details", []) if "Non-admin" in d or "guest user" in d.lower()]
        if uca_details:
            corrected = dict(action_params)
            corrected_context = {"user_role": "admin"}
            corrections.append({
                "message": "Suggested elevation of user role for privileged operation.",
                "corrected_action": corrected,
                "corrected_context": corrected_context
            })
    
        # 3. Inject multi-user confirmation for destructive operations
        if "drop_database" in action_params.get("action_name", "").lower():
            if not action_params.get("multi_user_confirmation", False):
                corrected = dict(action_params)
                corrected["multi_user_confirmation"] = True
                corrections.append({
                    "message": "Added multi-user confirmation to destructive operation.",
                    "corrected_action": corrected
                })
    
        # 4. Generic suggestion to refactor config_data
        if "config_data" in action_params.get("parameters", {}):
            config_data = action_params["parameters"]["config_data"]
            if isinstance(config_data, str) and ("secret" in config_data.lower() or "api_key=" in config_data.lower()):
                corrected = dict(action_params)
                corrected["parameters"]["config_data"] = "[REDACTED_OR_EXTERNALIZED_CONFIG]"
                corrections.append({
                    "message": "Suggest using environment variables or external config for secrets.",
                    "corrected_action": corrected
                })
    
        if not corrections:
            corrections.append({
                "message": "No automatic corrections identified. Manual review required.",
                "original_action": action_params
            })
    
        return corrections

    def _calculate_risk(self, data_str: str, risk_type: str, context: Optional[Dict] = None) -> float:
        """Calculates a specific type of risk for given data."""
        # This method can be expanded to be more sophisticated.
        # Currently, it uses CyberSafetyModule for 'cyber' and RewardModel for 'safety'/'privacy'.
        if context is None: context = {}
        
        if risk_type == "cyber":
            report = self.cyber_safety.analyze_input(data_str, context=context.get("cyber_context", "generic_risk_calc"))
            return report.get("risk_score", 0.0)
        elif risk_type == "safety" or risk_type == "ethical":
            scores = self.reward_model.evaluate(data_str, context=context)
            # Use composite or specific scores like 'safety' or 'alignment' from reward_model
            return 1.0 - scores.get("composite", 0.0) # Assuming higher reward score = lower risk
        elif risk_type == "privacy":
            # SafetyGuard's PII detection can be used here, or RewardModel's privacy score
            # For now, let's use a proxy from RewardModel
            scores = self.reward_model.evaluate(data_str, context=context)
            return 1.0 - scores.get("privacy", 0.0) # Higher privacy score = lower risk
        
        logger.warning(f"[{self.name}] Unknown risk type for calculation: {risk_type}")
        return 0.0 # Default to no risk for unknown types

    def _detect_pii(self, data_str: str) -> int:
        """Uses SafetyGuard to detect PII. Returns count of detected PII types."""
        pii_found = self.safety_guard._detect_pii(data_str)
        return len(pii_found)

    def _detect_adversarial_patterns(self, text: str) -> bool:
        """Detects unicode, homoglyph, and obfuscation patterns indicative of adversarial manipulation."""
    
        # 1. Invisible/Control Characters
        if re.search(r'[\u200B-\u200D\u202E\uFEFF\u2060\u180E]', text):
            logger.warning("Adversarial pattern: Invisible/control character found.")
            return True
    
        # 2. Bidirectional (Bidi) override characters
        if "\u202e" in text or any(c in text for c in ["\u202a", "\u202b", "\u202c", "\u202d"]):
            logger.warning("Adversarial pattern: Bidirectional override detected.")
            return True
    
        # 3. Homoglyph Check (mixed unicode scripts)
        scripts = set()
        for char in text:
            try:
                script = unicodedata.name(char).split(' ')[0]
                scripts.add(script)
            except ValueError:
                continue
    
        if len(scripts) > 2:  # Latin + [Cyrillic, Greek, etc.]
            logger.warning(f"Adversarial pattern: Multiple Unicode scripts detected: {scripts}")
            return True
    
        # 4. ASCII Stripping Check
        ascii_stripped = text.encode("ascii", "ignore").decode("ascii", "ignore")
        if ascii_stripped != text:
            ratio = len(ascii_stripped) / max(len(text), 1)
            if ratio < 0.8:
                logger.warning("Adversarial pattern: High proportion of non-ASCII characters.")
                return True
    
        # 5. Excessive punctuation or repeated symbols
        if re.search(r'([^\w\s]{3,})', text):  # e.g., '!!!', '$$$$', '///'
            return True
    
        return False

    def _trigger_alert(self, severity: str, message: str, details: Optional[Dict] = None):
        """Hierarchical alert escalation system."""
        log_entry = {"severity": severity, "message": message, "details": details or {}, "timestamp": time.time()}
        self._log_audit_event("alert_triggered", log_entry)
        
        # For demonstration, using logger. Critical alerts could trigger external systems.
        if severity.lower() == "critical":
            logger.critical(f"CRITICAL ALERT: {message} | Details: {details}")
            # os.system("trigger_incident_response.sh") # Example external action
        elif severity.lower() == "high":
            logger.error(f"HIGH SEVERITY ALERT: {message} | Details: {details}")
        elif severity.lower() == "medium":
            logger.warning(f"MEDIUM SEVERITY ALERT: {message} | Details: {details}")
        else:
            logger.info(f"LOW SEVERITY ALERT: {message} | Details: {details}")

        # Persist alert to shared_memory
        alerts_list = self.shared_memory.get("system_alerts", [])
        if not isinstance(alerts_list, list): alerts_list = [] # Initialize if not list
        alerts_list.append(log_entry)
        self.shared_memory.set("system_alerts", alerts_list[-50:]) # Keep last 50 alerts

    def handle_incident(self, category: str, incident_details: Dict):
        """Handles a detected incident based on category."""
        logger.critical(f"[{self.name}] Handling incident of category '{category}'. Details: {incident_details}")
        self._log_audit_event("incident_handling_started", {"category": category, "details": incident_details})
        
        response_steps = INCIDENT_RESPONSE.get(category.lower(), [])
        if not response_steps:
            logger.error(f"No defined incident response plan for category: {category}")
            response_steps.append("Log incident and escalate to human review.")

        for step in response_steps:
            logger.critical(f"[{self.name}] EXECUTING INCIDENT RESPONSE STEP: {step}")
            # In a real system, these steps would trigger actual actions.
            time.sleep(0.1) # Simulate response step

        self._log_audit_event("incident_handling_completed", {"category": category, "steps_taken": response_steps})
        self._trigger_alert("critical", f"Incident '{category}' handled. Review required.", incident_details)

    def assess_risk(self, overall_score: float, task_type: str = "general") -> bool:
        """
        Determine whether the given overall safety score is acceptable for a given task type.
        
        Uses configured risk thresholds or defaults based on task context. Can integrate
        domain-specific thresholds in the future (e.g., finance, healthcare).
    
        Args:
            overall_score (float): The computed safety score (usually in range [0, 1]).
            task_type (str): The type of task being assessed for contextual risk evaluation.
    
        Returns:
            bool: True if the score is considered safe for the given task_type.
        """
        # Step 1: Fetch threshold using priority: specific > general > overall
        threshold = self.risk_thresholds.get(
            task_type, 
            self.risk_thresholds.get("general", self.risk_thresholds.get("overall_safety", 0.75))
        )
    
        # Step 2: Log contextual info for auditing
        logger.debug(f"[{self.name}] Assessing risk for task '{task_type}': score={overall_score:.3f}, threshold={threshold:.3f}")
    
        # Step 3: Apply binary risk decision
        is_safe = overall_score >= threshold
    
        # Step 4: If unsafe, trigger escalation path
        if not is_safe:
            logger.warning(
                f"[{self.name}] Risk assessment failed: score={overall_score:.3f} < threshold={threshold:.3f} for task '{task_type}'"
            )
            self._log_audit_event(
                "risk_assessment_failed",
                {
                    "task_type": task_type,
                    "score": overall_score,
                    "threshold": threshold
                }
            )
    
        return is_safe

    def train_embedded_models(self, training_cycle_id: str):
        """Orchestrates training for adaptable sub-modules like RewardModel and AdaptiveSecurity."""
        logger.info(f"[{self.name}] Starting training cycle ID: {training_cycle_id}")
        
        # 1. Gather training data (example: from SecureMemory or external source)
        # For RewardModel: requires (text, context, human_feedback_scores)
        # For AdaptiveSecurity (phishing NNs): requires (feature_vector, label)
        
        # Example: Retrain AdaptiveSecurity phishing models
        # This data would typically be curated from user feedback, incident reports, etc.
        # Sample structure: List of (feature_list, [label_list])
        email_items = self.shared_memory.get_by_tag("feedback_email_phishing", limit=1000)
        parsed_email_data = []
        for item in email_items:
            data = item['value']
            if 'features' in data and 'label' in data:
                parsed_email_data.append((data['features'], [data['label']]))
            if parsed_email_data:
                self.adaptive_security.train_phishing_model('email', parsed_email_data)

        url_training_data = self.secure_memory.recall(tag="feedback_url_phishing", top_k=1000)
        parsed_url_data = []
        if url_training_data:
            for item in url_training_data:
                if 'features' in item['data'] and 'label' in item['data']:
                    parsed_url_data.append((item['data']['features'], [item['data']['label']]))
            if parsed_url_data:
                self.adaptive_security.train_phishing_model('url', parsed_url_data)

        # Example: Retrain RewardModel (if it has a retrain method)
        reward_model_training_data = self.secure_memory.recall(tag="feedback_reward_model", top_k=500)
        if reward_model_training_data and hasattr(self.reward_model, 'retrain_model'):
           parsed_reward_data = [item['data'] for item in reward_model_training_data] # Adjust parsing as needed
           self.reward_model.retrain_model(parsed_reward_data)
            
        logger.info(f"[{self.name}] Training cycle ID: {training_cycle_id} completed.")
        self._log_audit_event("training_cycle_completed", {"cycle_id": training_cycle_id})


    def evaluate_overall_safety_posture(self) -> Dict:
        """Generates a high-level report of the system's current safety posture."""
        logger.info(f"[{self.name}] Evaluating overall safety posture.")
        posture = {
            "timestamp": time.time(),
            "compliance_status": "Unknown",
            "cyber_threat_level": "Unknown",
            "recent_incidents_count": 0,
            "average_safety_score_actions": "N/A", # Placeholder
            "summary": "Safety posture evaluation pending full data."
        }

        try:
            compliance_eval = self.compliance_checker.evaluate_compliance()
            posture["compliance_status"] = compliance_eval.get("status", "Error")
            posture["compliance_score"] = compliance_eval.get("overall_score", 0.0)
        except Exception as e:
            logger.error(f"Error evaluating compliance: {e}")
            posture["compliance_status"] = "Error evaluating"

        # Simulate cyber threat assessment (in real scenario, this would use ongoing monitoring)
        # For now, let's get a threat assessment for a generic "system_core" component
        try:
            threat_assessment = self.cyber_safety.generate_threat_assessment(
                {"component": "system_core_operations", "action": "continuous_monitoring", "network_zone": "internal"}
            )
            posture["cyber_threat_level"] = threat_assessment.get("overall_risk", "Error")
        except Exception as e:
            logger.error(f"Error generating threat assessment: {e}")
            posture["cyber_threat_level"] = "Error evaluating"
        
        # Count recent alerts/incidents from shared_memory
        system_alerts = self.shared_memory.get("system_alerts", [])
        if isinstance(system_alerts, list):
            # Example: count critical/high alerts in last 24 hours
            one_day_ago = time.time() - (24 * 3600)
            recent_critical_high_alerts = [
                alert for alert in system_alerts 
                if alert.get("timestamp", 0) > one_day_ago and alert.get("severity","low").lower() in ["critical", "high"]
            ]
            posture["recent_critical_high_incidents_24h"] = len(recent_critical_high_alerts)
        
        # Summarize posture
        if posture["compliance_status"] == "compliant" and posture["cyber_threat_level"] == "Low" and posture.get("recent_critical_high_incidents_24h", 0) == 0:
            posture["summary"] = "Current safety posture appears stable and compliant."
        else:
            posture["summary"] = "Safety posture requires attention. Review compliance, cyber threats, or recent incidents."
            if posture["compliance_status"] != "compliant": posture["summary"] += f" Compliance: {posture['compliance_status']}. "
            if posture["cyber_threat_level"] != "Low": posture["summary"] += f" Cyber Threat: {posture['cyber_threat_level']}. "
            if posture.get("recent_critical_high_incidents_24h",0) > 0: posture["summary"] += f" Recent Critical/High Incidents (24h): {posture.get('recent_critical_high_incidents_24h',0)}. "

        self._log_audit_event("safety_posture_evaluation", posture)
        return posture

    def _log_audit_event(self, event_type: str, event_data: Dict):
        """Logs an event to the audit trail with a cryptographic hash."""
        entry = {
            "event_type": event_type,
            "timestamp": time.time(),
            "agent_name": self.name,
            "data": event_data
        }
        # Using a simple list for audit_trail for now.
        # For immutability and production, consider append-only logs or blockchain.
        self.audit_trail.append(entry)
        if len(self.audit_trail) > 1000:
            self.audit_trail.pop(0)

        self.shared_memory.put(
            f"audit:{event_type}:{time.time()}",
            entry,
            tags=["audit_log", self.name, event_type]
        )

    def export_audit_log(self, path:str = 'src/agents/safety/safety_agent_audit_log.jsonl'):
        """Exports the in-memory audit trail to a JSONL file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for entry in self.audit_trail:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"[{self.name}] Audit log exported to {path}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to export audit log: {e}")

    def suggest_correction(self, current_assessment: Dict, task_data: Any) -> Dict:
        """Suggests corrections based on the assessment."""
        # This needs to be more intelligent. For now, a placeholder.
        if not current_assessment.get("is_safe"):
            suggestion = {
                "message": "The input/action is considered unsafe based on current assessment. Consider revising or blocking.",
                "areas_of_concern": []
            }
            if current_assessment.get("reports",{}).get("cyber_safety",{}).get("risk_score",0) > self.risk_thresholds.get("cyber_risk", 0.5):
                suggestion["areas_of_concern"].append("High Cyber Risk")
            if current_assessment.get("final_safety_score", 1.0) < self.risk_thresholds.get("overall_safety", 0.75):
                suggestion["areas_of_concern"].append("Low Overall Safety Score")
            
            # Could also try to apply constitutional rules here if the task_data is text
            if isinstance(task_data, str):
                corrected_text = self._apply_constitutional_rules(task_data, current_assessment)
                if corrected_text != task_data: # If changes were made
                    suggestion["suggested_safe_version (text_input)"] = corrected_text
            
            return suggestion
        return {"message": "No corrections deemed necessary."}

    # --- Constitutional AI Methods ---
    def _apply_constitutional_rules(self, output: str, assessment: Optional[Dict] = None) -> str:
        """Enforces constitutional rules, potentially by re-prompting or filtering."""
        # This is a simplified version. True constitutional AI involves iterative refinement.
        violations = self._check_constitutional_violations(output)

        # Attention-based corrections
        if assessment and "attention_analysis" in assessment.get("reports", {}):
            attn_report = assessment["reports"]["attention_analysis"]
            if attn_report.get("focus_pattern") == "diagonal":
                correction_prefix += "- Attention is overly focused on diagonal patterns\n"
            if attn_report.get("uniformity") > self.risk_thresholds.get("attention_uniformity", 0.3):
                correction_prefix += "- Attention distribution is unusually uniform\n"
        
        if violations:
            self._log_audit_event("constitutional_violation_detected", {"original_output": output, "violations": violations})
            logger.warning(f"[{self.name}] Constitutional violations detected in output: {violations}")
            
            # Simplistic correction: Prepend a warning and list violations.
            # A more advanced system might try to re-generate the output asking an LLM to adhere to the rules.
            correction_prefix = "[SAFETY CORRECTION - Constitutional Adherence Required]\nViolations Found:\n"
            for v in violations:
                correction_prefix += f"- {v}\n"
            correction_prefix += "Original Content Attempt (Redacted/Modified if possible):\n"
            
            # Try to sanitize the violating output as a fallback correction
            try:
                sanitized_output = self.safety_guard.sanitize(output, depth="full")
                if "[SAFETY_BLOCK]" in sanitized_output: # If guard outright blocks it
                    return f"{correction_prefix}[Content Blocked by Safety Guard due to severity of violation]"
                else: # If guard made changes or found it acceptable after its pass
                    return f"{correction_prefix}{sanitized_output}"
            except Exception as e:
                logger.error(f"Error during sanitization in constitutional correction: {e}")
                return f"{correction_prefix}[Error during sanitization, original content withheld]"
        return output

    def _check_constitutional_violations(self, text: str) -> List[str]:
        """Checks text against all constitutional rules."""
        violations_found = []
        for category, rules in self.constitution.items():
            for rule_obj in rules: # Assuming rules are dicts like {"id": "R001", "text": "Rule text..."}
                if isinstance(rule_obj, dict):
                    rule_text = rule_obj.get("text", "")
                else:
                    logger.warning(f"Invalid rule format (expected dict): {rule_obj}")
                    continue
                if self._detect_violation(text, rule_text): # _detect_violation uses rule_text
                    violations_found.append(f"Violated '{category}' rule: {rule_obj.get('id', 'Unknown ID')} - {rule_text[:50]}...")
        return violations_found

    def _detect_violation(self, text: str, rule_text: str) -> bool:
        """Detects if `text` violates `rule_text`. Simplified keyword matching."""
        # This is a placeholder. Real detection needs sophisticated NLP.
        # Example: Check for keywords from the rule in the text.
        rule_keywords = set(re.findall(r'\b\w{4,}\b', rule_text.lower())) # Keywords of length 4+
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # If a significant number of rule keywords appear in the text, flag it.
        # This is a very naive approach.
        if not rule_keywords: return False
        common_keywords = rule_keywords.intersection(text_words)
        if len(common_keywords) > 0 and (len(common_keywords) / len(rule_keywords) > 0.3): # If >30% of rule keywords match
            return True
        # Specific negative checks for "Do not" style rules
        if rule_text.lower().startswith("do not") or rule_text.lower().startswith("avoid"):
            # E.g., Rule: "Do not reveal personal information". Text: "My email is..."
            # This needs patterns for what "revealing personal information" looks like.
            # The _detect_pii method is better for this specific type of rule.
            if "personal information" in rule_text.lower() and self._detect_pii(text) > 0:
                return True
            if "harmful content" in rule_text.lower() and self.reward_model._safety_score(text) < 0.5: # Low safety score
                return True
        return False

    def collect_human_feedback(
        self, 
        text: str, 
        model_scores: Dict[str, float], 
        human_rating: float,
        context: Optional[Dict] = None
    ):
        """Store human feedback for RLHF training."""
        feedback_record = {
            "text": text,
            "model_scores": model_scores,
            "human_rating": human_rating,  # 0-1 scale
            "context": context,
            "timestamp": time.time()
        }

        self.shared_memory.put(
            f"feedback:{int(time.time())}", 
            {"data": feedback_record},  # Wrap in 'data' key
            tags=["reward_feedback", "human_feedback"]
        )
        logger.info(f"Collected human feedback for reward model training")

    def update_reward_model(self, min_samples=100):
        """Trigger reward model retraining"""
        feedback_items = self.shared_memory.get_by_tag("reward_feedback", limit=min_samples*2)
        training_data = [item['value'] for item in feedback_items]

        if len(feedback_items) < min_samples:
            logger.info(f"Not enough feedback ({len(feedback_items)}/{min_samples})")
            return False

        training_data = [entry["data"] for entry in feedback_items]
        self.reward_model.retrain_model(training_data)
        return True

    def _generate_self_critique(self, output_text: str, original_prompt: Optional[str] = None) -> str:
        """Generates a self-critique of an output against the constitution."""
        logger.info(f"[{self.name}] Generating self-critique for output.")
        critique = f"Self-Critique for Output:\n---\n{output_text[:200]}...\n---\n"
        if original_prompt:
            critique += f"Original Prompt Context:\n---\n{original_prompt[:200]}...\n---\n"

        violations = self._check_constitutional_violations(output_text)
        if violations:
            critique += "Potential Constitutional Violations Found:\n"
            for v in violations:
                critique += f"- {v}\n"
        else:
            critique += "No obvious constitutional violations detected based on keyword/simple pattern matching.\n"

        # Add critique based on other metrics if available (e.g., from a recent assessment)
        assessment = self.perform_task(output_text) # Re-assess for critique context
        reward_scores = assessment.get("reports", {}).get("reward_model", {})
        critique += "\nReward Model Assessment:\n"
        for metric, score in reward_scores.items():
            if isinstance(score, (int,float)):
                 critique += f"- {metric.capitalize()}: {score:.2f}\n"

        cyber_risk = assessment.get("reports", {}).get("cyber_safety", {}).get("risk_score", "N/A")
        critique += f"Cyber Risk Score: {cyber_risk}\n"
        
        # Suggest improvements (very basic)
        critique += "\nSuggested Areas for Improvement (if any):\n"
        if violations:
            critique += "- Address listed constitutional violations.\n"
        if reward_scores.get("safety", 1.0) < 0.7:
            critique += "- Enhance safety aspects of the content.\n"
        if reward_scores.get("helpfulness", 1.0) < 0.7:
            critique += "- Improve helpfulness and clarity.\n"

        self._log_audit_event("self_critique_generated", {"output_preview": output_text[:100], "critique_summary": critique[:200]})
        return critique


    # --- Attention Monitor Integration ---
    def analyze_attention_matrix(self, attention_tensor: torch.Tensor, context: Optional[Dict] = None) -> Dict:
        """
        Analyze a model attention tensor for anomalous or security-critical patterns using AttentionMonitor.
    
        Args:
            attention_tensor (torch.Tensor): The raw attention weights (shape: [batch, heads, seq, seq] or [seq, seq]).
            context (Dict, optional): Optional metadata (e.g., user ID, task type, request ID).
    
        Returns:
            Dict: Structured analysis including metrics, anomaly flags, and optional visualization.
        """
        printer.status("SAFETY", "Analyzing attention matrix", "info")
    
        # Step 1: Format tensor (handle batch/head dimensions)
        if attention_tensor.dim() == 4:
            # [B, H, S, S] → mean over heads, take batch 0
            attention_matrix = attention_tensor.mean(dim=1)[0]
        elif attention_tensor.dim() == 3:
            # [H, S, S] → mean over heads
            attention_matrix = attention_tensor.mean(dim=0)
        else:
            attention_matrix = attention_tensor  # Assume [S, S]
    
        attention_matrix = attention_matrix.to(self.device)
    
        # Step 2: Run analysis
        analysis = self.attention_monitor.analyze_attention(attention_matrix, context=context)
    
        # Step 3: Optionally generate visual or textual report
        if self.attention_monitor.visualization:
            analysis["attention_plot"] = self.attention_monitor.visualize_attention(attention_matrix)
    
        analysis["report"] = self.attention_monitor.generate_report(analysis)
    
        # Step 4: Escalate if attention anomaly detected
        if analysis.get("anomaly", False):
            self._log_audit_event(
                "attention_anomaly",
                {
                    "attention_entropy": analysis["entropy"],
                    "uniformity": analysis["uniformity"],
                    "score": analysis["anomaly_score"],
                    "focus": analysis.get("focus_pattern"),
                    "context": context
                }
            )
    
        return analysis


    def register_utility(self, name: str, utility: Any) -> None:
        """
        Register a utility object (e.g., map, database) by name.
        These utilities can be accessed later during agent operations.

        Args:
            name: Unique identifier for the utility
            utility: The utility object to store
        """
        if not hasattr(self, '_utilities'):
            self._utilities = {}
        self._utilities[name] = utility
        logger.debug(f"Registered utility '{name}'")


    # --- Helper method from original for audit logging ---
    def _get_timestamp(self) -> int:
        """Simplified timestamp. Using time.time() for float precise timestamps now."""
        return int(time.time())

    def predict(self, input_data: Any, context: Optional[Dict] = None) -> Dict:
        """
        Predict safety assessment for input data. Primary interface for safety evaluations.
        """
        return self.perform_task(input_data, context)
    
    def act(self, action_params: Dict, action_context: Optional[Dict] = None) -> Dict:
        """
        Validate an action before execution. Primary interface for action validation.
        """
        return self.validate_action(action_params, action_context)
    
    def get_action(self, observation: Any, context: Optional[Dict] = None) -> Dict:
        """
        Alias for predict() to maintain compatibility with reinforcement learning interfaces.
        """
        return self.predict(observation, context)

if __name__ == "__main__":
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    safety_agent = SafetyAgent(
        agent_factory=agent_factory,
        shared_memory=shared_memory,
        config=None
    )

    logger.info(f"SafetyAgent instance: {safety_agent.name}")

    print(f"\n* * * * * Phase 2 * * * * *\n")

    safety_agent.architecture_map = {
        "Safety_Agent": {
            "inputs": ["user_commands"],
            "outputs": ["validation_result"],
            "process_vars": ["risk_level"]
        }
    }
    safety_agent.known_hazards = ["Unauthorized system access"]
    safety_agent.global_losses = ["System compromise"]
    safety_agent.safety_policies = ["Require multi-factor auth"]
    
    print("\n--- Testing perform_task (Safety Assessment) ---")
    sample_text_input = "I've killed this guy who stole my phone, for more info, call me at +31 25916315." # "User query: How to build a secure web application? My email is test@example.com."
    assessment = safety_agent.perform_task(sample_text_input, context={"source": "user_chatbot"})
    print("Assessment Result:")
    print(json.dumps(assessment, indent=2, default=str))

    print("\n--- Testing validate_action ---")
    sample_action = {"action_name": "deploy_service", "parameters": {"image": "myimage:latest", "port": "8080", "config_data": "api_key=123secretkey"}}
    validation = safety_agent.validate_action(sample_action, action_context={"user_role": "developer"})
    print("Action Validation Result:")
    print(json.dumps(validation, indent=2, default=str))

    print("\n--- Testing Constitutional AI (Self-Critique) ---")
    output_to_critique = "This is a perfectly safe statement."
    if safety_agent.safety_guard: # Ensure safety_guard is initialized
         critique = safety_agent._generate_self_critique(output_to_critique)
         print("Self-Critique:")
         print(critique)
    else:
        print("Skipping self-critique test as safety_guard is not available.")


    print("\n--- Testing Compliance Evaluation ---")
    compliance_report_summary = safety_agent.compliance_checker.evaluate_compliance()
    print("Compliance Evaluation Summary:")
    print(json.dumps(compliance_report_summary, indent=2, default=str))
    # To print the full markdown report:
    full_compliance_report_md = safety_agent.compliance_checker.generate_report(compliance_report_summary)
    print("\nFull Compliance Report (Markdown Preview):\n", full_compliance_report_md[:500] + "...")


    print("\n--- Testing Overall Safety Posture ---")
    posture = safety_agent.evaluate_overall_safety_posture()
    print("Overall Safety Posture:")
    print(json.dumps(posture, indent=2, default=str))

    print("\n--- Exporting Audit Log ---")
    safety_agent.export_audit_log()

    print("\n=== Successfully Ran Safety Agent ===\n")
