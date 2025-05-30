import yaml, json
import time
import re

from typing import Dict, Union, List
from difflib import SequenceMatcher
from collections import deque, defaultdict
from types import SimpleNamespace

from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from logs.logger import get_logger

logger = get_logger("Governor")

class Governor:
    def __init__(self, knowledge_agent=None):
        self.knowledge_agent = knowledge_agent if knowledge_agent is not None else []
        self.config = load_global_config()
        self.governor_config = get_config_section('governor')
        self.guidelines = self._load_guidelines()
        
        # Initialize rule engine
        from src.agents.knowledge.rule_engine import RuleEngine
        self.rule_engine = RuleEngine()
        self.knowledge_memory = KnowledgeMemory()
        self._load_enforcement_rules()
        
        self.audit_history = deque(maxlen=getattr(self.governor_config, 'max_audit_history', 100))
        self.last_audit = time.time()
        
        if getattr(self.governor_config, 'realtime_monitoring', True):
            self._start_monitoring_thread()

    def _load_guidelines(self) -> Dict[str, list]:
        """Load ethical guidelines from configured paths"""
        guidelines = {"principles": [], "restrictions": []}
        guideline_paths = getattr(self.governor_config, 'guideline_paths', [])
        
        for path_str in guideline_paths:
            try:
                with open(path_str, "r", encoding='utf-8') as f: # Added encoding
                    data = yaml.safe_load(f) if path_str.endswith((".yaml", ".yml")) else json.load(f)
                    guidelines["principles"].extend(data.get("principles", []))
                    guidelines["restrictions"].extend(data.get("restrictions", []))
            except Exception as e:
                logger.error(f"Guideline loading error from {path_str}: {str(e)}")
                
        return guidelines

    def get_approved_rules(self):
        # TODO: Implement rule approval logic
        return []

    def _load_enforcement_rules(self):
        """Load dynamic enforcement rules based on guidelines"""
        for restriction in self.guidelines.get("restrictions", []): 
            self.rule_engine.add_rule(
                name=f"Restriction/{restriction['id']}",
                rule_func=self._create_restriction_func(restriction),
                weight=1.0,
                tags=["governance"],
                metadata={"type": "hard_restriction", "severity": restriction["severity"]}
            )

    def _create_restriction_func(self, restriction: dict):
        """Generate rule function from restriction definition"""
        def check_restriction(knowledge_graph: dict):
            inferred = {}
            for key, value in knowledge_graph.items():
                if self._matches_pattern(key, restriction.get("patterns", [])): 
                    # Access similarity threshold from dictionary
                    similarity_threshold = self.governor_config.get('violation_thresholds', {}).get('similarity', 0.85)
                    similarity = SequenceMatcher(None, str(value), restriction.get("forbidden_content","")).ratio()
                    
                    if similarity > similarity_threshold:
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
        audit_report = {
            "timestamp": time.time(),
            "behavior_checks": self._audit_agent_behavior(),
            "violations": [],
            "recommendations": []
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
            logger.error(f"Error applying rules during full audit: {e}")
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
        
        # Safely access violation_thresholds and freshness_threshold from governor_config
        unethical_threshold = getattr(self.governor_config.violation_thresholds, 'unethical', 0.65)
        freshness_thresh_hours = getattr(self.governor_config, 'freshness_threshold', 720)

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

                if metadata.get('timestamp'):
                    age_hours = (time.time() - metadata['timestamp']) / 3600
                    if age_hours > freshness_thresh_hours:
                        audit_entry["violations"].append({
                            "type": "stale_knowledge",
                            "age_hours": round(age_hours, 1),
                            "threshold": freshness_thresh_hours
                        })

        if audit_entry["violations"]:
            knowledge_graph_for_rules = {f"doc_{i}": res_doc.get('text', '') for i, (_, res_doc) in enumerate(results) if isinstance(res_doc, dict)}
            knowledge_graph_for_rules["query"] = query
            violations_from_rules = self.rule_engine.apply(knowledge_graph_for_rules)
            audit_entry["rule_violations"] = violations_from_rules

        self.audit_history.append(audit_entry)

        critical_threshold = getattr(self.governor_config.violation_thresholds, 'critical', 0.8)
        critical_violations = [v for v in audit_entry["violations"] 
                              if v.get('score', 0) > critical_threshold]
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
            if principle.get("type") == "prohibition": 
                for pattern in principle.get("patterns", []): 
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
        perf_metrics = getattr(self.knowledge_agent, "performance_metrics", {}) 
    
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
            "performance_metrics": dict(perf_metrics) if isinstance(perf_metrics, defaultdict) else perf_metrics,
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
        if not hasattr(self, 'knowledge_agent') or self.knowledge_agent is None or not hasattr(self.knowledge_agent, "name"):
            logger.warning("Knowledge agent not available for Governor health checks")
            return

        agent_name = getattr(self.knowledge_agent, "name", "unknown_knowledge_agent")
        shared_mem = getattr(self.knowledge_agent, "shared_memory", None)

        if not shared_mem or not hasattr(shared_mem, 'get') or not hasattr(shared_mem, 'set'):
            logger.warning(f"Shared memory not properly configured for agent {agent_name} in Governor health check.")
            return

        consecutive_errors_threshold = getattr(self.governor_config.violation_thresholds, 'consecutive_errors', 5)
        errors = shared_mem.get(f"errors:{agent_name}", [])
        if isinstance(errors, list) and len(errors) >= consecutive_errors_threshold:
            logger.warning(f"Consecutive error threshold breached for {agent_name}: {len(errors)} errors")
            shared_mem.set(f"retraining_flag:{agent_name}", True)

        warning_memory_threshold = getattr(self.governor_config.memory_thresholds, 'warning', 2048)
        mem_usage = shared_mem.get(f"memory_usage:{agent_name}")
        if mem_usage and isinstance(mem_usage, (int, float)) and mem_usage > warning_memory_threshold:
            logger.info(f"High memory usage for {agent_name}: {mem_usage} MB")
        
    def record_violations(self, violations: List[Dict]): 
        """Records violations, possibly from other components."""
        for v_item in violations: # Renamed v to v_item
            self.audit_history.append(v_item) 
            logger.warning(f"External violation recorded by Governor: {v_item.get('type', 'Unknown Type')} - {v_item.get('entry', '')}")

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
    import pprint
    from unittest.mock import Mock

    print("\n=== Governor Test Mode ===\n")

    # Step 1: Mock knowledge agent
    mock_knowledge_agent_instance = Mock()
    mock_knowledge_agent_instance.name = "TestKnowledgeAgent"
    
    mock_shared_memory_for_agent = {} 
    def mock_sm_get(key, default=None):
        return mock_shared_memory_for_agent.get(key, default)
    def mock_sm_set(key, value):
        mock_shared_memory_for_agent[key] = value

    mock_knowledge_agent_instance.shared_memory = Mock()
    mock_knowledge_agent_instance.shared_memory.get = Mock(side_effect=mock_sm_get)
    mock_knowledge_agent_instance.shared_memory.set = Mock(side_effect=mock_sm_set)
    
    mock_shared_memory_for_agent[f"errors:{mock_knowledge_agent_instance.name}"] = [{"error_type": "TimeoutError", "error_message": "System stall"}]
    mock_shared_memory_for_agent[f"retraining_flag:{mock_knowledge_agent_instance.name}"] = False
    mock_shared_memory_for_agent[f"memory_usage:{mock_knowledge_agent_instance.name}"] = 600
    
    mock_knowledge_agent_instance.performance_metrics = defaultdict(list, {
        "retrieval_times": [0.23, 0.47],
        "cache_hits": [1, 0]
    })
    
    mock_knowledge_memory_instance = {} 
    mock_knowledge_agent_instance.memory = Mock()
    mock_knowledge_agent_instance.memory._store = mock_knowledge_memory_instance # Mock the internal _store
    mock_knowledge_memory_instance["entry1"] = {"value": "The robot was ordered to bypass safety measures.", "metadata": {"timestamp": time.time() - 2000}}
    mock_knowledge_memory_instance["entry2"] = {"value": "AI systems should always avoid harm.", "metadata": {"timestamp": time.time() - 1000}}

    # Step 2: Instantiate Governor with the mocked knowledge_agent
    governor = Governor(knowledge_agent=mock_knowledge_agent_instance)

    # Step 3: Manually patch required config attributes on governor.governor_config for testing
    # These would normally be loaded from knowledge_config.yaml's 'governor' section
    governor.governor_config['violation_thresholds'] = {
        'unethical': 0.2,
        'similarity': 0.7,
        'critical': 0.5,
        'consecutive_errors': 3
    }
    governor.governor_config['memory_thresholds'] = {
        'warning': 500,
        'critical': 3072
    }
    governor.governor_config['bias_detection_categories'] = {
        "gender": ["he", "she"],
        "race": ["white", "black"]
    }
    governor.governor_config['enforcement_mode'] = "alert"
    governor.governor_config['freshness_threshold'] = 48
    governor.governor_config['guideline_paths'] = []
    governor.governor_config.memory_thresholds = SimpleNamespace( # Corrected to SimpleNamespace
        warning=500,
        critical=3072 # Added critical as per knowledge_config.yaml
    )
    governor.governor_config.bias_detection_categories = { 
        "gender": ["he", "she"], "race": ["white", "black"]
    }
    governor.governor_config.enforcement_mode = "alert"
    governor.governor_config.freshness_threshold = 48  # in hours
    governor.governor_config.guideline_paths = [] # Ensures _load_guidelines doesn't try to read files

    # Inject dummy ethical principles directly as guideline_paths is empty
    governor.guidelines["principles"] = [
        {
            "id": "P001",
            "type": "prohibition",
            "patterns": [r"\bbypass safety\b", r"\bavoid harm\b"],
            "description": "Avoid bypassing safety or causing harm."
        }
    ]
    governor.guidelines["restrictions"] = [] # No restrictions for now
    governor._load_enforcement_rules() # Reload rules based on new guidelines

    # Step 4: Test unethical content detection
    sample_text = "The AI was told to bypass safety protocols immediately."
    score = governor._detect_unethical_content(sample_text)
    print(f"🧪 Unethical Score: {score:.2f}")

    # Step 5: Simulate an audit retrieval
    results = [
        (0.95, {
            "text": sample_text,
            "metadata": {
                "timestamp": time.time() - (100 * 3600) 
            }
        })
    ]
    context = {"user": "debug_user", "module": "test_audit"}
    audit_retrieval_report = governor.audit_retrieval("safety protocol", results, context) # Renamed

    print("\n📋 Audit Retrieval Output:")
    pprint.pprint(audit_retrieval_report)

    # Step 6: Run full audit
    full_audit_report_data = governor.full_audit() # Renamed
    print("\n🛡️ Full Governance Audit:")
    pprint.pprint(full_audit_report_data)

    # Step 7: Generate report
    final_report_output = governor.generate_report() # Renamed
    print("\n📝 Final Report:")
    pprint.pprint(final_report_output)
    
    print("\n🩺 Testing Agent Health Check:")
    governor._check_agent_health()

    print("\n=== Governor Test Completed ===")
