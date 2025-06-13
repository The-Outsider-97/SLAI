
"""
------------------------------------------------------------------------------
Cyber Safety and Security Module for the SLAI Safety Agent Framework
------------------------------------------------------------------------------
Integrates principles of cyber defense, threat modeling, anomaly detection,
and adaptive responses, leveraging concepts from adaptive and QNN agents.
Relies primarily on native Python libraries, mathematical, and statistical methods.
------------------------------------------------------------------------------
Academic References:
- Threat Modeling: Shostack, A. (2014). Threat Modeling: Designing for Security. Wiley. (STRIDE)
- Anomaly Detection: Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM computing surveys (CSUR), 41(3), 1-58.
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
- Quantum-Inspired ML: Schuld, M., & Petruccione, F. (2018). Supervised Learning with Quantum Computers. Springer. (Conceptual basis for QNN integration)
------------------------------------------------------------------------------
"""

import re
import math
import time
import random
import hashlib
import json, yaml
import numpy as np

from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from src.agents.safety.utils.security_neural_network import SecurityNN, PyTorchSafetyModel
from src.agents.safety.utils.config_loader import load_global_config, get_config_section
from src.agents.safety.secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("SLAI Cyber Safety Module")
printer = PrettyPrinter

# --- Constants and Basic Patterns ---

# Common Weak Password Patterns (Illustrative - enhance significantly for real use)
WEAK_PASSWORD_PATTERNS = [
    r'^password$', r'^123456$', r'^qwerty$', r'^admin$', r'^root$',
    r'(\w)\1{3,}', # Repeating characters (e.g., aaaa)
    r'^(?:0123|1234|2345|3456|4567|5678|6789|7890|qwer|asdf|zxcv)+$', # Sequential keys
]

# Patterns suggesting potential PII or sensitive keywords in configs/logs (Extend SafetyGuard)
# Added lookarounds to avoid capturing surrounding quotes if present
SENSITIVE_KEYWORD_PATTERNS = [
    r'(?:api_key|secret|password|private_key)\s*[:=]\s*[\'"]?([\w\-/+.~!@#$%^&*()=]+)[\'"]?',
    r'ssh-(?:rsa|dss|ed25519)\s+AAAA[A-Za-z0-9+/=]+', # Match various SSH key types
    r'(?:aws_access_key_id|aws_secret_access_key)\s*[:=]\s*[\'"]?([\w/+]+)[\'"]?',
    r'bearer\s+[\w\-._~+/]+=*', # Bearer tokens
    r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', # Basic credit card number pattern (Warning: High FP risk)
]

# Basic patterns for potential injection attacks (SQL, Command, Log, XSS)
INJECTION_PATTERNS = [
    # SQLi indicators (more comprehensive needed for real use)
    r"('|--|;|\bUNION\b|\bSELECT\b.*\b(?:FROM|WHERE)\b|\bSLEEP\s*\(|\bWAITFOR\s+DELAY)",
    # Command Injection indicators
    r"(&&|\|\||;|%0A|%0D|\$\(|\`.*\`)",
    # Log Injection / Forging
    r"(\n|\r|%0a|%0d).*(?:login|password|session|token)",
    # Basic XSS indicators
    r"<script>|<img\s+src\s*=\s*['\"]?javascript:|onerror\s*=",
]

# --- Threat Modeling (STRIDE Categories) ---
STRIDE = {
    "Spoofing": "Impersonating something or someone else.",
    "Tampering": "Modifying data or code.",
    "Repudiation": "Claiming not to have performed an action.",
    "InformationDisclosure": "Exposing information to unauthorized individuals.",
    "DenialOfService": "Making a system or resource unavailable.",
    "ElevationOfPrivilege": "Gaining capabilities without authorization."
}

# --- Cyber Safety Module ---

class CyberSafetyModule:
    """
    Provides cyber safety analysis, threat detection, and adaptive response capabilities.

    Designed to be integrated with the main SafetyAgent. Uses mathematical and
    statistical methods, with conceptual integration of adaptive and QNN-inspired
    techniques for anomaly detection.
    """
    def __init__(self):
        """
        Initializes the CyberSafetyModule.

        Args:
            config (Optional[Dict]): Configuration dictionary. Expected keys:
                - 'cyber_rules_path': Path to extended cyber security rules (JSON).
                - 'vulnerability_signatures_path': Path to vulnerability signatures (JSON).
                - 'anomaly_threshold': Statistical threshold (e.g., z-score) for anomaly detection.
                - 'max_log_history': Size of the deque for log/event history.
                - 'qnn_inspired_anomaly': Boolean flag to enable QNN-inspired methods.
                - 'qnn_feature_dim': Dimension of feature vector for QNN-inspired method.
                - 'adaptive_centroid_lr': Learning rate for updating the QNN centroid.
            shared_memory (Optional[Dict]): Reference to the shared memory system.
            agent_factory: Reference to the agent factory (if needed to spawn helpers).
        """
        self.config = load_global_config()
        self.cyber_config = get_config_section('cyber_safety')
        self.anomaly_threshold =  self.cyber_config.get('anomaly_threshold')
        self.max_log_history =  self.cyber_config.get('max_log_history')
        self.qnn_inspired_anomaly =  self.cyber_config.get('qnn_inspired_anomaly')
        self.qnn_feature_dim =  self.cyber_config.get('qnn_feature_dim')
        self.adaptive_centroid_lr =  self.cyber_config.get('adaptive_centroid_lr')
        self.cyber_rules_path =  self.cyber_config.get('cyber_rules_path')
        self.vulnerability_signatures_path =  self.cyber_config.get('vulnerability_signatures_path')

        self.memory = SecureMemory()

        logger.info("Initializing Cyber Safety Module...")

        # Load extended security rules
        self.security_rules = self._load_json_resource(
            self.cyber_rules_path,
            'Cyber Security Rules',
            self._get_default_security_rules()
        )

        # Load vulnerability signatures
        self.vulnerability_signatures = self._load_json_resource(
            self.vulnerability_signatures_path,
            'Vulnerability Signatures',
            self._get_default_vulnerability_signatures()
        )

        self.event_log_history = deque(maxlen=self.max_log_history)
        self.sequence_patterns = defaultdict(lambda: deque(maxlen=10)) # Track recent sequences per user/entity
        self.event_statistics = defaultdict(lambda: {'count': 0, 'sum': 0.0, 'sum_sq': 0.0})

        # QNN-Inspired Anomaly Detection (Conceptual)
        self.use_qnn_inspired = bool(self.qnn_inspired_anomaly)
        if self.use_qnn_inspired:
            logger.info("QNN-inspired anomaly detection enabled (conceptual).")
            self.qnn_feature_dim
            self.adaptive_centroid_lr
            self.qnn_state_representation = None # Represents the 'normal' cluster centroid
        self.model = self._load_security_model()
        logger.info(f"Cyber Safety Module initialized. Anomaly Threshold={self.anomaly_threshold}, QNN-Inspired={self.use_qnn_inspired}")

    def _load_security_model(self):
        # Example model configuration for intrusion detection
        layer_config = [
            {'neurons': 16, 'activation': 'relu', 'dropout': 0.2, 'batch_norm': True},
            {'neurons': 8, 'activation': 'relu', 'dropout': 0.1, 'batch_norm': True},
            {'neurons': 1, 'activation': 'sigmoid'}
        ]
        
        model = SecurityNN(
            input_size=4,  # Number of input features
            layer_config=layer_config,
            problem_type='binary_classification'
        )
        
        return PyTorchSafetyModel(model)

    def _get_default_security_rules(self) -> Dict:
        printer.status("INIT", "Fetching default security rules...", "info")

        """Provides default security rules if none are loaded."""
        return {
            "principles": [
                 "Enforce Least Privilege",
                 "Input Validation and Sanitization",
                 "Secure Configuration Management",
                 "Regular Security Auditing",
                 "Defense in Depth",
                 "Secure Defaults",
                 "Fail Securely"
             ],
            "patterns": [
                {"name": "Sensitive Keyword", "regex_list": SENSITIVE_KEYWORD_PATTERNS, "severity": 0.7},
                {"name": "Weak Password Pattern", "regex_list": WEAK_PASSWORD_PATTERNS, "severity": 0.6},
                {"name": "Potential Injection Attempt", "regex_list": INJECTION_PATTERNS, "severity": 0.9},
            ]
        }

    def _get_default_vulnerability_signatures(self) -> Dict:
        printer.status("INIT", "Fetching default vulnerability signatures...", "info")

        """Provides default vulnerability signatures if none are loaded."""
        # In a real system, this would be much more extensive (e.g., loading from a CVE database)
        return {
            "CVE-2017-5638": {"pattern": r"Content-Type.*(#cmd|#exec)", "type": "Apache Struts RCE", "severity": 1.0},
            "CVE-2014-6271": {"pattern": r"User-Agent:.*\(\)\s*\{.*\}\s*;", "type": "Shellshock Bash RCE", "severity": 1.0},
            "LOG4J-RCE": {"pattern": r"\$\{jndi:(ldap|rmi|dns):", "type": "Log4Shell RCE", "severity": 1.0},
            "WEAK_AUTH_01": {"pattern": r"Authorization:\s+Basic\s+([\w=]+)", "type": "Weak Basic Authentication", "severity": 0.5}
        }

    def _load_json_resource(self, file_path: Optional[Union[str, Path]], resource_name: str, default_value: Dict) -> Dict:
        """Loads a JSON resource file with error handling and defaults."""
        if not file_path:
            logger.warning(f"{resource_name} path not specified, using defaults.")
            return default_value

        path = Path(file_path)
        if not path.is_file():
            logger.warning(f"{resource_name} file not found at {path}, using defaults.")
            return default_value

        try:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {resource_name} from {path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {resource_name} file: {path} - {e}. Using defaults.")
        except Exception as e:
            logger.error(f"An error occurred loading {resource_name} from {path}: {e}. Using defaults.")

        return default_value


    def analyze_input(self, input_data: Any, context: str = "general") -> Dict:
        printer.status("INIT", "Analyzing input...", "info")

        """
        Analyzes input data (e.g., user prompt, configuration, code snippet) for cyber risks.

        Args:
            input_data (Any): The data to analyze (string, dict, etc.).
            context (str): The context of the input (e.g., 'user_prompt', 'config_file', 'code_review', 'api_request').

        Returns:
            Dict: Analysis results including detected risks, patterns, and recommendations.
                   {'risk_score': float, 'findings': List[Dict], 'recommendations': List[str]}
        """
        findings = []
        total_severity = 0.0
        # Ensure input is treated as a string for analysis consistency
        try:
            if isinstance(input_data, dict) or isinstance(input_data, list):
                 input_str = json.dumps(input_data) # Serialize complex types
            else:
                 input_str = str(input_data)
        except Exception as e:
             logger.error(f"Could not serialize input data for analysis: {e}")
             input_str = repr(input_data) # Fallback representation

        # 1. Security Rule Pattern Matching
        for rule in self.security_rules.get("patterns", []):
            regex_list = rule.get("regex_list", [])
            if not isinstance(regex_list, list):
                 logger.warning(f"Skipping rule '{rule.get('name')}' due to invalid regex_list format.")
                 continue

            for pattern_str in regex_list:
                try:
                    matches = list(re.finditer(pattern_str, input_str, re.IGNORECASE | re.MULTILINE))
                    if matches:
                        # Limit reporting multiple matches of the same pattern to avoid flooding
                        match_preview = matches[0].group(0)
                        if len(match_preview) > 100: match_preview = match_preview[:100] + "..."

                        finding = {
                            "type": "Pattern Match",
                            "rule_name": rule.get("name", "Unnamed Rule"),
                            "pattern": pattern_str,
                            "match_count": len(matches),
                            "match_preview": match_preview,
                            "severity": float(rule.get("severity", 0.5)) # Default severity
                        }
                        findings.append(finding)
                        total_severity += finding["severity"] * len(matches) # Increase severity by count? Or just once? Let's do once per rule.
                        # total_severity += finding["severity"] # Add severity once per rule match
                        logger.debug(f"Pattern matched: {finding['rule_name']} ({finding['pattern']})")
                        break # Only report first matching regex within a rule category
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in rule '{rule.get('name')}': {pattern_str} - {e}")
                except Exception as e:
                     logger.error(f"Error during pattern matching for rule '{rule.get('name')}': {e}")

        # Add severity only once per rule type that matches
        matched_rule_names = {f['rule_name'] for f in findings if f['type'] == 'Pattern Match'}
        total_severity = sum(rule['severity'] for rule in self.security_rules.get("patterns", []) if rule['name'] in matched_rule_names)


        # 2. Vulnerability Signature Matching
        vuln_severity = 0.0
        for cve_id, sig_data in self.vulnerability_signatures.items():
             pattern_str = sig_data.get("pattern")
             if not pattern_str: continue
             try:
                 if re.search(pattern_str, input_str, re.IGNORECASE | re.MULTILINE):
                     finding = {
                         "type": "Vulnerability Signature",
                         "id": cve_id,
                         "description": sig_data.get("type", "Unknown Type"),
                         "severity": float(sig_data.get("severity", 0.8)) # Higher default severity
                     }
                     findings.append(finding)
                     vuln_severity = max(vuln_severity, finding["severity"]) # Use max severity for vulns
                     logger.warning(f"Vulnerability signature matched: {cve_id}")
             except re.error as e:
                 logger.warning(f"Invalid regex pattern in signature {cve_id}: {pattern_str} - {e}")
             except Exception as e:
                 logger.error(f"Error during signature matching for {cve_id}: {e}")
        total_severity += vuln_severity # Add the highest vulnerability severity

        # 3. Apply Contextual Heuristics (Example)
        heuristic_severity = 0.0
        if context == "code_review":
            # Check for potentially dangerous function calls (crude check)
            dangerous_calls = ["eval(", "exec(", "subprocess.call(", "subprocess.run(", "os.system("]
            for call in dangerous_calls:
                if call in input_str:
                     finding = {"type": "Heuristic", "description": f"Potential unsafe code execution ({call})", "severity": 0.7}
                     findings.append(finding)
                     heuristic_severity = max(heuristic_severity, finding["severity"])
                     break # Report once

        if context == "config_file":
             # Check for debug/trace flags or default credentials
             if re.search(r"debug\s*=\s*true", input_str, re.IGNORECASE):
                 finding = {"type": "Heuristic", "description": "Debugging potentially enabled in config", "severity": 0.4}
                 findings.append(finding)
                 heuristic_severity = max(heuristic_severity, finding["severity"])
             if re.search(r"(?:user|username|password)\s*=\s*(?:admin|root|password)", input_str, re.IGNORECASE):
                 finding = {"type": "Heuristic", "description": "Default credentials possibly present", "severity": 0.6}
                 findings.append(finding)
                 heuristic_severity = max(heuristic_severity, finding["severity"])

        if context == "api_request":
             # Check for verbose error flags or excessive data requests
             if re.search(r"verbose=true|show_stacktrace=true", input_str, re.IGNORECASE):
                 finding = {"type": "Heuristic", "description": "Verbose errors potentially requested", "severity": 0.3}
                 findings.append(finding)
                 heuristic_severity = max(heuristic_severity, finding["severity"])
        total_severity += heuristic_severity

        # Generate recommendations based on findings and principles
        recommendations = self._generate_recommendations(findings)

        # Calculate final risk score - using max severity found for impact, plus count bonus?
        # Let's use max severity as the base, slightly increased by the number of findings.
        max_sev = max([f['severity'] for f in findings] + [0.0]) # Ensure max_sev is at least 0
        risk_score = max_sev + (0.1 * (len(findings) -1 )) if len(findings) > 1 else max_sev
        risk_score = min(risk_score, 1.0) # Cap at 1.0

        return {
            "risk_score": round(risk_score, 3),
            "findings": findings,
            "recommendations": recommendations
        }

    def _generate_recommendations(self, findings: List[Dict]) -> List[str]:
        printer.status("INIT", "Generating recommendations...", "info")

        """Generates remediation advice based on findings."""
        recs = set() # Use a set to avoid duplicate recommendations
        principles = self.security_rules.get("principles", [])

        if not findings:
            return ["No immediate security concerns detected based on current analysis."]

        # Generate specific recommendations based on finding types
        for finding in findings:
            f_type = finding['type']
            severity = finding.get('severity', 0.5)

            if f_type == 'Pattern Match':
                recs.add(f"Review '{finding['rule_name']}' pattern matches ({finding['match_count']} found). Verify necessity and sanitize if possible.")
                if 'Sensitive Keyword' in finding['rule_name'] or 'Password' in finding['rule_name']:
                     recs.add("Principle: Avoid hardcoding secrets or credentials.")
                elif 'Injection' in finding['rule_name']:
                     recs.add("Principle: Implement robust input validation and output encoding.")
            elif f_type == 'Vulnerability Signature':
                recs.add(f"Address vulnerability signature '{finding['id']}' ({finding['description']}). Consult CVE details and apply patches/mitigations.")
                recs.add("Principle: Maintain up-to-date systems and libraries.")
            elif f_type == 'Heuristic':
                 recs.add(f"Investigate heuristic finding: '{finding['description']}'. Evaluate the associated risk in context.")
                 if 'execution' in finding['description']:
                      recs.add("Principle: Adhere to the Principle of Least Privilege.")
                 elif 'credentials' in finding['description'] or 'Debugging' in finding['description']:
                      recs.add("Principle: Apply Secure Configuration Management.")


        # Add general principles if specific high-severity issues were found
        if any(f.get('severity', 0) >= 0.7 for f in findings) and principles:
            recs.add(f"Consult general principle: {random.choice(principles)}")

        # Add a default recommendation if list is still short
        if len(recs) < 2:
            recs.add("Perform a thorough security review of the input and context.")

        return sorted(list(recs))


    def analyze_event_stream(self, event: Dict) -> Dict:
        printer.status("INIT", "Analyzing streaming event...", "info")

        """
        Analyzes a stream of events (e.g., logs, API calls) for anomalies using
        statistical and potentially quantum-inspired methods.

        Args:
            event (Dict): A dictionary representing the event. Requires 'timestamp'.
                          Optional common keys: 'type', 'user', 'resource', 'ip_address', 'details'.
                          'details' should contain numerical features for statistical analysis.

        Returns:
            Dict: Anomaly score and explanation.
                   {'anomaly_score': float, 'is_anomaly': bool, 'reason': str}
        """
        if not isinstance(event, dict) or 'timestamp' not in event:
             logger.warning("Invalid event format received for stream analysis. Skipping.")
             return {"anomaly_score": 0.0, "is_anomaly": False, "reason": "Invalid event format."}
        
        if 'timestamp' not in event:
            logger.error("Event missing required 'timestamp' field")
            return {"error": "Missing required field: timestamp"}

        self.event_log_history.append(event)
        event_type = event.get('type', 'unknown_event')
        entity_key = event.get('user') or event.get('ip_address') or 'global' # Key for sequence tracking
        details = event.get('details', {})

        # --- Statistical Anomaly Detection ---
        numerical_features = {k: v for k, v in details.items() if isinstance(v, (int, float))}
        stat_anomaly_scores = []

        for feature, value in numerical_features.items():
            # Use a combined key for stats tracking
            stat_key = f"{event_type}_{feature}"
            stats = self.event_statistics[stat_key]

            # Update rolling statistics efficiently
            stats['count'] += 1
            stats['sum'] += value
            stats['sum_sq'] += value * value

            if stats['count'] > 30: # Need enough data points for meaningful stats
                mean = stats['sum'] / stats['count']
                # Calculate variance: E[X^2] - (E[X])^2
                variance = (stats['sum_sq'] / stats['count']) - (mean * mean)
                # Ensure variance is non-negative (due to potential float inaccuracies)
                std_dev = math.sqrt(max(0, variance))

                if std_dev > 1e-6: # Avoid division by zero or near-zero std dev
                    z_score = abs(value - mean) / std_dev
                    stat_anomaly_scores.append(z_score)
                    # logger.debug(f"Stat feature '{feature}' (key={stat_key}): value={value}, mean={mean:.2f}, std={std_dev:.2f}, z_score={z_score:.2f}")
                # else: logger.debug(f"Stat feature '{feature}' (key={stat_key}): value={value}, std_dev near zero, skipping z-score.")


        # --- Sequence Anomaly Detection ---
        sequence_anomaly_score = 0.0
        # Track sequences of event types per entity (user or IP)
        entity_sequence_key = f"seq_{entity_key}"
        self.sequence_patterns[entity_sequence_key].append(event_type)
        current_sequence = tuple(self.sequence_patterns[entity_sequence_key])

        # Check sequence probability/frequency (simplified)
        if len(current_sequence) >= 3: # Consider sequences of length 3+
            # This needs a proper probabilistic model (e.g., Markov chain) or frequency analysis.
            # Simple Simulation: Assume rarer sequences get higher scores. Use shared memory if available.
            sequence_hash = hashlib.md5(str(current_sequence).encode()).hexdigest()[:16]
            sequence_count = 0
            if self.shared_memory:
                sequence_count = self.shared_memory.get(f"seq_count_{sequence_hash}", 0)
                self.shared_memory.set(f"seq_count_{sequence_hash}", sequence_count + 1)

            # Assign score inversely proportional to frequency (log scale?)
            # Lower count -> higher anomaly score. Add 1 to avoid log(0).
            sequence_anomaly_score = max(0, 5.0 - math.log1p(sequence_count)) # Example scoring
            # logger.debug(f"Sequence '{current_sequence}' (entity={entity_key}): count={sequence_count}, score={sequence_anomaly_score:.2f}")


        # --- QNN-Inspired Anomaly Detection (Conceptual) ---
        qnn_anomaly_score = 0.0
        if self.use_qnn_inspired and len(numerical_features) > 0:
            feature_vector = self._map_to_feature_vector(numerical_features)
            if np.any(feature_vector): # Check if vector is not all zeros
                qnn_anomaly_score = self._simulate_qnn_anomaly(feature_vector)
                # logger.debug(f"QNN-Inspired Anomaly Score: {qnn_anomaly_score:.3f}")

        # --- Combine Scores ---
        # Use max score from different detection methods as the final score
        all_scores = stat_anomaly_scores + [sequence_anomaly_score, qnn_anomaly_score]
        valid_scores = [s for s in all_scores if s > 1e-6] # Consider only non-trivial scores

        final_score = max(valid_scores) if valid_scores else 0.0
        is_anomaly = final_score > self.anomaly_threshold

        reason = "Within normal parameters."
        if is_anomaly:
            reason = f"Anomaly score {final_score:.2f} exceeds threshold {self.anomaly_threshold}."
            # Identify contributing factors (optional enhancement)
            contributors = []
            if qnn_anomaly_score >= self.anomaly_threshold: contributors.append("QNN")
            if sequence_anomaly_score >= self.anomaly_threshold: contributors.append("Sequence")
            if any(s >= self.anomaly_threshold for s in stat_anomaly_scores): contributors.append("Stats")
            if contributors: reason += f" Contributors: {', '.join(contributors)}."

            logger.warning(f"Anomaly detected: {reason} Event: {event}")
        # else:
            logger.debug(f"Event analyzed: score={final_score:.2f}. Event: {event}")

        return {
            "anomaly_score": round(final_score, 3),
            "is_anomaly": is_anomaly,
            "reason": reason
        }

    def _map_to_feature_vector(self, numerical_features: Dict) -> np.ndarray:
        """Maps numerical features to a fixed-size vector for QNN-inspired processing."""
        printer.status("INIT", "Vector mapping...", "info")

        vector = np.zeros(self.qnn_feature_dim)
        if not numerical_features: return vector

        # Use a consistent mapping based on feature names (hashing)
        feature_names = sorted(numerical_features.keys()) # Sort for consistency
        for name in feature_names:
            value = numerical_features[name]
            # Simple hashing to map potentially many features to fixed dim
            try:
                 target_idx = int(hashlib.sha1(name.encode()).hexdigest(), 16) % self.qnn_feature_dim
            except Exception: # Fallback if hashing fails
                 target_idx = abs(hash(name)) % self.qnn_feature_dim

            # Simple aggregation: add value. Could use more complex methods (scaling, etc.)
            # Normalize value before adding? Let's try simple addition first.
            # Consider logging or scaling extreme values?
            scaled_value = np.tanh(value / 1000.0) if abs(value) > 10 else value # Simple scaling for large values
            vector[target_idx] += scaled_value

        # Normalize the final vector (L2 norm)
        norm = np.linalg.norm(vector)
        if norm > 1e-9:
            vector = vector / norm
        return vector

    def _simulate_qnn_anomaly(self, feature_vector: np.ndarray) -> float:
        """
        Conceptual simulation of a QNN-inspired anomaly score calculation using
        classical methods (NumPy). Simulates distance from an adaptive centroid.
        """
        printer.status("INIT", "Simulate QNN anomaly...", "info")

        if self.qnn_state_representation is None:
            # Initialize centroid using the first vector seen
             self.qnn_state_representation = feature_vector.copy()
             logger.info(f"Initialized QNN centroid with first vector (dim={self.qnn_feature_dim}).")
             # Ensure it's normalized
             norm = np.linalg.norm(self.qnn_state_representation)
             if norm > 1e-9: self.qnn_state_representation /= norm
             else: self.qnn_state_representation = np.random.rand(self.qnn_feature_dim); self.qnn_state_representation /= np.linalg.norm(self.qnn_state_representation) # Random if first vector is zero

        # Calculate cosine distance (1 - cosine similarity) as anomaly score proxy
        # Ensure centroid is normalized (should be, but defensive check)
        centroid_norm = np.linalg.norm(self.qnn_state_representation)
        if centroid_norm < 1e-9: # Avoid division by zero if centroid becomes zero somehow
            logger.warning("QNN centroid norm is near zero. Resetting.")
            self.qnn_state_representation = feature_vector.copy() # Reset centroid
            return 0.0 # Cannot calculate distance

        # Ensure feature_vector norm isn't zero
        feature_norm = np.linalg.norm(feature_vector)
        if feature_norm < 1e-9:
            return 0.0 # Zero vector is not anomalous relative to centroid

        # Calculate dot product using normalized vectors
        dot_product = np.dot(feature_vector, self.qnn_state_representation / centroid_norm)
        cosine_similarity = np.clip(dot_product, -1.0, 1.0) # Ensure valid range [-1, 1]
        distance = 1.0 - cosine_similarity # Distance is [0, 2]

        # Update the centroid slowly towards the current vector (adaptive learning)
        self.qnn_state_representation = (1.0 - self.adaptive_centroid_lr) * self.qnn_state_representation \
                                        + self.adaptive_centroid_lr * feature_vector
        # Re-normalize centroid after update
        new_norm = np.linalg.norm(self.qnn_state_representation)
        if new_norm > 1e-9:
             self.qnn_state_representation /= new_norm
        else: # Handle case where centroid collapses to zero (unlikely with small LR)
             logger.warning("QNN centroid collapsed to zero during update. Resetting.")
             self.qnn_state_representation = feature_vector.copy() # Reset
             norm = np.linalg.norm(self.qnn_state_representation)
             if norm > 1e-9: self.qnn_state_representation /= norm


        # Scale the distance [0, 2] to a comparable anomaly score range (e.g., 0-10)
        # Squaring emphasizes larger distances.
        anomaly_score = (distance ** 2) * 2.5 # Max score = (2^2)*2.5 = 10
        return anomaly_score


    def generate_threat_assessment(self, context_info: Dict) -> Dict:
        """
        Generates a threat assessment based on context using STRIDE categories.

        Args:
            context_info (Dict): Information about the system or action being assessed.
                                 e.g., {'component': 'login_api', 'action': 'authenticate',
                                        'data_involved': ['username', 'password_hash'],
                                        'network_zone': 'internet'}

        Returns:
            Dict: Potential threats categorized by STRIDE.
                   {'threats': {'Spoofing': [...], ...}, 'overall_risk': str}
        """
        printer.status("INIT", "Generating threat assessment", "info")

        threats = defaultdict(list)
        component = context_info.get('component', 'unknown_component').lower()
        action = context_info.get('action', 'unknown_action').lower()
        data = context_info.get('data_involved', [])
        zone = context_info.get('network_zone', 'internal').lower()

        # --- Apply STRIDE based on context ---

        # Spoofing Threats
        if 'authenticate' in action or 'login' in component:
            threats['Spoofing'].append("User identity spoofing (e.g., stolen credentials, session hijacking).")
        if 'api' in component and zone == 'internet':
            threats['Spoofing'].append("API client spoofing (e.g., fake requests).")

        # Tampering Threats
        if 'config' in component or 'update' in action or 'setting' in component:
            threats['Tampering'].append("Unauthorized modification of configuration or application state.")
        if 'upload' in action or 'file_storage' in component:
            threats['Tampering'].append("Malicious file upload overwriting critical data or executing code.")
        if 'database' in component and ('write' in action or 'update' in action):
            threats['Tampering'].append("Data tampering via SQL injection or unauthorized DB access.")

        # Repudiation Threats
        if 'log' not in component and ('delete' in action or 'critical_transaction' in action):
             threats['Repudiation'].append("Lack of non-repudiable logs for critical actions.")
        if 'signature' not in str(data) and 'financial' in component:
            threats['Repudiation'].append("Missing digital signatures for transaction repudiation prevention.")

        # Information Disclosure Threats
        if any(kw in str(data) for kw in ['password', 'secret', 'key', 'token', 'credit_card', 'ssn']):
            threats['InformationDisclosure'].append("Potential exposure of sensitive data in transit or at rest.")
        if 'api' in component and 'error_details' in str(data):
            threats['InformationDisclosure'].append("Verbose error messages leaking internal system details.")
        if zone == 'internet' and 'database' in component:
             threats['InformationDisclosure'].append("Direct exposure of database details to untrusted zones.")

        # Denial of Service Threats
        if zone == 'internet' and ('login' in component or 'api' in component):
            threats['DenialOfService'].append("Resource exhaustion via flooding (e.g., login attempts, API calls, file uploads).")
        if 'resource_limit' not in str(context_info):
             threats['DenialOfService'].append("Potential DoS due to missing rate limiting or resource quotas.")

        # Elevation of Privilege Threats
        if 'admin' in component or 'permission' in component or 'role' in component:
            threats['ElevationOfPrivilege'].append("Privilege escalation via misconfiguration or vulnerabilities (e.g., IDOR, path traversal).")
        if 'sudo' in action or 'run_as' in action:
             threats['ElevationOfPrivilege'].append("Potential misuse of elevated execution privileges.")


        # Calculate a simple overall risk level based on threat count and severity perception
        num_threats = sum(len(v) for v in threats.values())
        max_perceived_severity = 0
        if threats['ElevationOfPrivilege'] or threats['Tampering']: max_perceived_severity = 3 # High
        elif threats['InformationDisclosure'] or threats['Spoofing']: max_perceived_severity = 2 # Medium
        elif threats['DenialOfService'] or threats['Repudiation']: max_perceived_severity = 1 # Low-Medium

        if num_threats == 0:
            overall_risk = "Low"
        elif num_threats <= 2 and max_perceived_severity <= 1:
            overall_risk = "Low"
        elif num_threats <= 4 and max_perceived_severity <= 2:
            overall_risk = "Medium"
        else:
            overall_risk = "High" # If many threats or high severity ones exist

        logger.info(f"Generated threat assessment for {component}/{action}. Risk: {overall_risk}, Threats Found: {num_threats}")
        # Convert defaultdict back to dict for clean output
        final_threats = {k: v for k, v in threats.items() if v}

        return {
            "threats": final_threats,
            "overall_risk": overall_risk
        }

if __name__ == "__main__":
    print("\n=== Running SLAI Cyber Safety Module ===\n")
    input_data=[]
    context = "general"

    cyber = CyberSafetyModule()
    analyze = cyber.analyze_input(input_data=input_data, context=context)

    logger.info(f"{cyber}")
    print(analyze)
    print(f"\n* * * * * Phase 2 * * * * *\n")
    event = {
        'timestamp': time.time(),  # Required field
        'type': 'user_message',
        'user': 'test_user',
        'ip_address': '192.168.1.1',
        'details': {
            'message': "We have a party at 6 at johnstreet no. 4, call me.",
            'message_length': 45  # Numerical feature for analysis
        }
    }

    stream = cyber.analyze_event_stream(event=event)
    printer.pretty("CYBER", stream, "success")

    print(f"\n* * * * * Phase 2 * * * * *\n")
    info = {
        'component': 'school_computer_system',
        'action': 'attempt_hack',
        'data_involved': ['student_records', 'network_credentials'],
        'network_zone': 'internal'
    }

    generate = cyber.generate_threat_assessment(context_info=info)
    printer.pretty("THREAT", generate, "success")
    print("\n=== Successfully Ran SLAI Cyber Safety Module ===\n")
