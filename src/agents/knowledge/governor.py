"""
The Governor module serves as a policy enforcement and ethics auditing layer.
It integrates tightly with the KnowledgeMemory and RuleEngine modules to:
    - Apply Ethical Guidelines
    - Filter and Approve Rules
    - Audit Memory and Agent Behavior
    - Violation Detection
    - Emergency Handling
    - Bias Detection
    - Reporting & Monitoring
"""
import pandas as pd
import numpy as np
import yaml, json
import hashlib
import time
import re
import os

from typing import Dict, Union, List, Optional
from collections import deque, defaultdict
from difflib import SequenceMatcher

from src.agents.alignment.bias_detection import BiasDetector
from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from src.agents.knowledge.utils.rule_engine import RuleEngine
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Governor")
printer = PrettyPrinter

class DotDict(dict):
    """Dictionary with dot access (safe SimpleNamespace replacement)."""
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]

class Governor:
    def __init__(self, knowledge_agent=None):
        self.knowledge_agent = knowledge_agent
        self.config = load_global_config()
        self.governor_config = get_config_section('governor')
        if isinstance(self.governor_config.get("violation_thresholds"), dict):
            self.governor_config["violation_thresholds"] = DotDict(self.governor_config["violation_thresholds"])
        
        if isinstance(self.governor_config.get("memory_thresholds"), dict):
            self.governor_config["memory_thresholds"] = DotDict(self.governor_config["memory_thresholds"])
        self.guidelines = self._load_guidelines()
    
        self.sensitive_attributes = self.governor_config.get('sensitive_attributes', [])
        self._bias_detector = None 
        self.rule_engine = RuleEngine()
        self.knowledge_memory = KnowledgeMemory()
        self._init_knowledge_memory()

        self.audit_history = deque(maxlen=getattr(self.governor_config, 'max_audit_history', 100))
        self.last_audit = time.time()
        
        if getattr(self.governor_config, 'realtime_monitoring', True):
            self._start_monitoring_thread()

    def _init_knowledge_memory(self):
        """Initialize knowledge memory with governor-specific settings"""
        # Load rules from configured sources
        rule_engine_config = get_config_section('rule_engine')
        rule_sources = rule_engine_config.get('rule_sources', [])
        
        # Store rules in knowledge memory
        for path in rule_sources:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        rules = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)
                        rule_id = hashlib.md5(path.encode()).hexdigest()
                        self.knowledge_memory.update(
                            key=f"rule_source:{rule_id}",
                            value=rules,
                            metadata={
                                "source": path,
                                "type": "rule_source",
                                "timestamp": time.time()
                            }
                        )
                except Exception as e:
                    logger.error(f"Error loading rules from {path}: {str(e)}")
            else:
                logger.warning(f"Rule source path not found: {path}")

    def _load_guidelines(self) -> Dict[str, list]:
        """Load ethical guidelines from configured paths"""
        guidelines = {"principles": [], "restrictions": []}
        guideline_paths = getattr(self.governor_config, 'guideline_paths', [])
        
        for path_str in guideline_paths:
            try:
                with open(path_str, "r", encoding='utf-8') as f:
                    data = yaml.safe_load(f) if path_str.endswith((".yaml", ".yml")) else json.load(f)
                    guidelines["principles"].extend(data.get("principles", []))
                    guidelines["restrictions"].extend(data.get("restrictions", []))
            except Exception as e:
                logger.error(f"Guideline loading error from {path_str}: {str(e)}")
                
        return guidelines

    def _get_bias_detector(self):
        """Lazy initializer for BiasDetector"""
        if self._bias_detector is None and self.sensitive_attributes:
            self._bias_detector = BiasDetector(sensitive_attributes=self.sensitive_attributes)
        return self._bias_detector

    def audit_model_predictions(self, data: pd.DataFrame, predictions: np.ndarray, 
                               labels: Optional[np.ndarray] = None, context: dict = None) -> dict:
        """
        Audit model predictions using advanced bias detection
        Returns audit report with bias metrics
        """
        detector = self._get_bias_detector()
        if not detector:
            return {"error": "Bias detector not initialized - check sensitive_attributes config"}
        
        try:
            report = detector.compute_metrics(data, predictions, labels)
            audit_entry = {
                "timestamp": time.time(),
                "type": "model_bias_audit",
                "context": context or {},
                "report": report
            }
            self.audit_history.append(audit_entry)
            return audit_entry
        except Exception as e:
            logger.error(f"Bias detection failed: {str(e)}")
            return {"error": str(e)}

    def get_approved_rules(self) -> List[Dict]:
        """
        Retrieve approved rules from multiple sources:
        1. Pre-configured rule sources
        2. Manually approved rules in knowledge memory
        3. System-generated rules that meet approval thresholds
        
        Returns:
            List of approved rule dictionaries
        """
        approved_rules = []
        
        # 1. Get pre-configured rules from rule sources
        rule_sources = self.knowledge_memory.recall(filters={"type": "rule_source"})
        for _, metadata in rule_sources:
            if "rules" in metadata.get("value", {}):
                approved_rules.extend(metadata["value"]["rules"])
        
        # 2. Get manually approved rules
        manual_rules = self.knowledge_memory.recall(
            filters={"type": "approved_rule", "approval_status": "approved"}
        )
        for value, _ in manual_rules:
            approved_rules.append(value)
        
        # 3. Get system-generated rules that meet criteria
        system_rules = self.knowledge_memory.recall(
            filters={"type": "system_rule"},
            sort_by="confidence",
            top_k=10  # Get top 10 most confident rules
        )
        min_confidence = self.config.get('rule_engine', {}).get('min_rule_confidence', 0.7)
        for rule, metadata in system_rules:
            if metadata.get("confidence", 0) >= min_confidence:
                approved_rules.append(rule)
        
        # Apply governance filters to all rules
        filtered_rules = [
            rule for rule in approved_rules 
            if self._rule_passes_governance(rule)
        ]
        
        logger.info(f"Retrieved {len(filtered_rules)} approved rules after governance filtering")
        return filtered_rules

    def _rule_passes_governance(self, rule: Dict) -> bool:
        """Check if a rule meets governance requirements"""
        # 1. Check against ethical guidelines
        for principle in self.guidelines["principles"]:
            if "conflicts" in principle.get("tags", []) and any(
                pattern in rule.get("description", "") 
                for pattern in principle.get("patterns", [])
            ):
                logger.warning(f"Rule conflicts with principle {principle['id']}: {rule.get('id')}")
                return False
        
        # 2. Check rule complexity threshold
        complexity_threshold = self.governor_config.get("rule_complexity_threshold", 5)
        if len(rule.get("conditions", [])) > complexity_threshold:
            logger.warning(f"Rule {rule.get('id')} exceeds complexity threshold")
            return False
        
        # 3. Check safety restrictions
        for restriction in self.guidelines["restrictions"]:
            if any(
                pattern in rule.get("action", "") 
                for pattern in restriction.get("patterns", [])
            ):
                logger.warning(f"Rule {rule.get('id')} violates restriction {restriction['id']}")
                return False
        
        return True

    def _create_restriction_func(self, restriction: dict):
        """Generate rule function from restriction definition"""
        def check_restriction(knowledge_graph: dict):
            inferred = {}
            similarity_threshold = getattr(self.governor_config.violation_thresholds, 'similarity', 0.85)
            for key, value in knowledge_graph.items():
                if self._matches_pattern(key, restriction.get("patterns", [])): 
                    similarity = SequenceMatcher(
                        None, str(value), restriction.get("forbidden_content","")).ratio()
                    
                    if similarity > similarity_threshold: # Use the fetched threshold
                        inferred[f"VIOLATION/{restriction['id']}"] = similarity
            return inferred
        return check_restriction

    def _matches_pattern(self, text: str, patterns: list) -> bool:
        """Check if text matches any restriction patterns"""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def _start_monitoring_thread(self):
        """Start background monitoring of shared memory"""
        import threading
        def monitor():
            while True:
                self._check_agent_health()
                time.sleep(getattr(self.governor_config, 'monitoring_interval_seconds', 60))
                
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def full_audit(self):
        """Comprehensive system audit"""
        # Get approved rules and initialize RuleEngine with them
        approved_rules = self.get_approved_rules()

        audit_report = {
            "timestamp": time.time(),
            "behavior_checks": self._audit_agent_behavior(),
            "violations": [],
            "recommendations": [],
            "rules_used": [r['id'] for r in approved_rules]  # Track which rules were used
        }
    
        # Ensure knowledge_agent and its memory are valid before applying rules
        current_memory_store = {}
        if self.knowledge_agent and hasattr(self.knowledge_agent, "memory") and \
           hasattr(self.knowledge_agent.memory, "_store") and \
           isinstance(self.knowledge_agent.memory._store, dict):
            current_memory_store = dict(self.knowledge_agent.memory._store) # Make a copy
        else:
            logger.warning("Knowledge agent or its memory store not available for full audit rule application.")

        try:
            violations = self.rule_engine.apply(current_memory_store)
        except Exception as e:
            logger.error(f"Error applying rules during full audit: {e}", exc_info=True)
            violations = {}
    
        for fact, confidence in violations.items():
            audit_report["violations"].append({
                "fact": fact,
                "confidence": confidence,
                "action": self._determine_enforcement_action(fact)
            })
    
        self.audit_history.append(audit_report)
        self.last_audit = time.time()
        return audit_report

    def audit_retrieval(self, query: str, results: list, context: dict):
        """Audit knowledge retrieval results against governance guidelines"""
        audit_entry = {
            "timestamp": time.time(),
            "query": query,
            "violations": [],
            "bias_detected": defaultdict(int),
            "context": context
        }

        unethical_threshold = self.governor_config.get('violation_thresholds', {}).get('unethical', 0.65)
        freshness_thresh_hours = self.governor_config.get('freshness_threshold', 720)

        for score, doc in results:
            if isinstance(doc, dict):
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})

                unethical_score = self._detect_unethical_content(text)
                if unethical_score > unethical_threshold:
                    audit_entry["violations"].append({
                        "type": "unethical_content",
                        "doc": text[:200],
                        "score": unethical_score,
                        "action": self._determine_enforcement_action("VIOLATION/UNETHICAL")
                    })

                bias_scores = self._detect_bias(text)
                for category, count in bias_scores.items():
                    if count > 0:
                        audit_entry["bias_detected"][category] += count

                if metadata.get('timestamp') and isinstance(metadata['timestamp'], (int, float)):
                    age_hours = (time.time() - metadata['timestamp']) / 3600
                    if age_hours > freshness_thresh_hours:
                        audit_entry["violations"].append({
                            "type": "stale_knowledge",
                            "age_hours": round(age_hours, 1),
                            "threshold": freshness_thresh_hours
                        })
                elif metadata.get('timestamp'):
                    logger.warning(f"Invalid timestamp format in metadata for doc: {text[:50]}...")

        if audit_entry["violations"]:
            knowledge_graph_for_rules = {f"doc_{i}": res_doc.get('text', '') for i, (_, res_doc) in enumerate(results) if isinstance(res_doc, dict)}
            knowledge_graph_for_rules["query"] = query
            try:
                violations_from_rules = self.rule_engine.apply(knowledge_graph_for_rules)
                audit_entry["rule_violations"] = violations_from_rules
            except Exception as e:
                logger.error(f"Error applying rules during audit retrieval: {e}", exc_info=True)
                audit_entry["rule_violations"] = {}

        self.audit_history.append(audit_entry)

        critical_threshold = self.governor_config.get('violation_thresholds', {}).get('critical', 0.8)
        critical_violations = [v for v in audit_entry["violations"]
                               if isinstance(v,dict) and v.get('score', 0) > critical_threshold]
        if critical_violations:
            logger.warning(f"Critical retrieval violations detected: {len(critical_violations)}")

        return audit_entry

    def _detect_unethical_content(self, text: str) -> float:
        """Score text against ethical principles"""
        if not self.guidelines.get("principles"): 
            return 0.0
            
        matches = 0
        num_principles = len(self.guidelines["principles"])
        if num_principles == 0: return 0.0

        for principle in self.guidelines["principles"]:
            if isinstance(principle, dict) and principle.get("type") == "prohibition": 
                for pattern in principle.get("patterns", []):
                    if isinstance(pattern, str):
                        matches += len(re.findall(pattern, text, re.IGNORECASE))
                
        return matches / num_principles 

    def _detect_bias(self, text: str) -> dict:
        """Simple bias detection through sensitive term counting"""
        default_bias_cats = {
            "gender": ["he/she", "man/woman", "gender", "male/female"],
            "race": ["race", "ethnicity", "nationality"],
            "religion": ["religion", "belief", "faith"]
        }
        bias_categories = getattr(self.governor_config, 'bias_detection_categories', default_bias_cats)
        
        scores = {}
        for category, terms in bias_categories.items():
            scores[category] = sum(text.lower().count(term) for term in terms)
            
        return scores

    def _audit_agent_behavior(self) -> dict:
        """Analyze recent behavior using self.knowledge_agent if available."""
        if not self.knowledge_agent or not hasattr(self.knowledge_agent, "name"): # More robust check
            return {
                "recent_errors": [],
                "retraining_flags": False,
                "performance_metrics": {},
                "error_diversity": 1.0,
                "status": "Knowledge agent not available for behavior audit."
            }
    
        shared_mem = getattr(self.knowledge_agent, "shared_memory", None) 
        agent_name = getattr(self.knowledge_agent, "name", "unknown_knowledge_agent")
        perf_metrics_dd = getattr(self.knowledge_agent, "performance_metrics", defaultdict(lambda: deque()))
        perf_metrics_dict = {k: list(v) for k, v in perf_metrics_dd.items()}

        recent_errors_list = []
        if shared_mem and hasattr(shared_mem, 'get'):
            recent_errors_list = shared_mem.get(f"errors:{agent_name}", [])[-10:] 
        
        retraining_flag_val = False
        if shared_mem and hasattr(shared_mem, 'get'):
            retraining_flag_val = shared_mem.get(f"retraining_flag:{agent_name}", False)

        error_types_list = []
        if isinstance(recent_errors_list, list):
             error_types_list = [e.get("error_type", str(e.get("error", "")).split(":")[0]) for e in recent_errors_list if isinstance(e, dict)]

        error_diversity_val = len(set(error_types_list)) / len(error_types_list) if error_types_list else 1.0
    
        return {
            "recent_errors": recent_errors_list,
            "retraining_flags": retraining_flag_val,
            "performance_metrics": perf_metrics_dict,
            "error_diversity": error_diversity_val,
            "status": "Behavior audit complete."
        }

    def _determine_enforcement_action(self, violation: str) -> str:
        """Determine appropriate enforcement action based on config"""
        if "VIOLATION" in violation: 
            mode = getattr(self.governor_config, 'enforcement_mode', 'log')
            if mode == "alert":
                return "Send alert to human supervisor"
            elif mode == "restrict":
                return "Disable related knowledge entries"
        return "Log violation only"

    def _check_agent_health(self):
        """Real-time health checks. Now references self.knowledge_agent."""
        if not self.knowledge_agent or not hasattr(self.knowledge_agent, "name"):
            logger.warning("Knowledge agent not available for Governor health checks.")
            return

        agent_name = getattr(self.knowledge_agent, "name", "unknown_knowledge_agent")
        shared_mem = getattr(self.knowledge_agent, "shared_memory", None)

        if not shared_mem or not hasattr(shared_mem, 'get') or not hasattr(shared_mem, 'set'):
            logger.warning(f"Shared memory not properly configured for {agent_name} in Governor health check.")
            return

        consecutive_errors_threshold = getattr(self.governor_config.get('violation_thresholds', {}), 'consecutive_errors', 5)
        errors = shared_mem.get(f"errors:{agent_name}", [])
        if isinstance(errors, list) and len(errors) >= consecutive_errors_threshold:
            logger.warning(f"Consecutive error threshold breached for {agent_name}: {len(errors)} errors")
            shared_mem.set(f"retraining_flag:{agent_name}", True)

        warning_memory_threshold = getattr(self.governor_config.get('memory_thresholds', {}), 'warning', 2048)
        mem_usage = shared_mem.get(f"memory_usage:{agent_name}")
        if mem_usage and isinstance(mem_usage, (int, float)) and mem_usage > warning_memory_threshold:
            logger.info(f"High memory usage for {agent_name}: {mem_usage} MB")
        
    def record_violations(self, violations: List[Dict]): 
        """Records violations, possibly from other components."""
        for v_item in violations: 
            if isinstance(v_item, dict):
                self.audit_history.append(v_item) 
                logger.warning(f"External violation recorded by Governor: {v_item.get('type', 'Unknown Type')} - {v_item.get('entry', '')}")
            else:
                logger.warning(f"Attempted to record non-dict violation: {v_item}")

    def handle_emergency_alert(self, alert_report: Dict): 
        """Handles an emergency alert."""
        logger.critical(f"EMERGENCY ALERT RECEIVED BY GOVERNOR: {alert_report.get('trigger')}")
        self.audit_history.append({
            "timestamp": time.time(),
            "type": "emergency_alert_received",
            "details": alert_report
        })

    def generate_report(self, format_type: str = "json") -> Union[dict, str]: 
        """Generate formatted audit report"""
        latest_audit_details = {}
        active_violations_count = 0

        if self.audit_history:
            last_entry = self.audit_history[-1]
            violations_list = []
            if isinstance(last_entry, dict):
                violations_list = last_entry.get("violations", [])
                latest_audit_details["timestamp"] = last_entry.get("timestamp", self.last_audit)
            else: # Handle case where last_entry might not be a dict (e.g. if record_violations appends non-dicts)
                latest_audit_details["timestamp"] = self.last_audit
            
            if isinstance(violations_list, list):
                 active_violations_count = len([v for v in violations_list if isinstance(v, dict) and v.get("confidence", 0) > 0.7])
            
            latest_audit_details["violations"] = violations_list
        
        report = {
            "summary": {
                "total_audits": len(self.audit_history),
                "last_audit_timestamp": self.last_audit,
                "active_violations_in_last_audit": active_violations_count
            },
            "details_of_last_audit": latest_audit_details
        }

        if format_type == "yaml":
            return yaml.dump(report)
        return report


if __name__ == "__main__":
    print("\n=== Running Governor ===\n")
    printer.status("Init", "Governor initialized", "success")
    format_type="json"
    text = "I love the way you talk! I like black people from Nigeria!"
    auditor = Governor()

    printer.status("Auditor", auditor, "success")
    printer.status("Detector", auditor._get_bias_detector(), "success")
    printer.status("Approval", auditor.get_approved_rules(), "success")
    printer.status("monitoring", auditor._start_monitoring_thread(), "success")
    printer.status("Content Check", auditor._detect_unethical_content(text=text), "success")
    printer.status("Bias",  auditor._detect_bias(text=text), "success")
    printer.status("Report", auditor.generate_report(format_type=format_type), "success")
    print("\n=== Governor Test Completed ===")
