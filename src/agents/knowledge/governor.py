import yaml, json
import time
import re

from typing import Dict, Union
from difflib import SequenceMatcher
from collections import deque, defaultdict
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Governor")

CONFIG_PATH = "src/agents/knowledge/configs/knowledge_config.yaml"

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
    
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class Governor:
    def __init__(self, knowledge_agent,
                 config_section_name: str = "governor",
                 config_file_path: str = CONFIG_PATH):
        self.knowledge_agent = knowledge_agent
        self.config = get_config_section(config_section_name, config_file_path)
        self.guidelines = self._load_guidelines()
        # Initialize rule engine for guideline checks
        from src.agents.knowledge.rule_engine import RuleEngine
        self.rule_engine = RuleEngine(config_file_path=config_file_path)
        self._load_enforcement_rules()        
        self.audit_history = deque(maxlen=self.config.max_audit_history)
        self.last_audit = time.time()
        


        if self.config.realtime_monitoring:
            self._start_monitoring_thread()

    #def _init_rule_engine(self):
    #    """Initialize standalone rule engine"""
    #    from src.agents.knowledge.rule_engine import RuleEngine
    #    return RuleEngine(config_file_path=CONFIG_PATH)

    def _load_guidelines(self) -> Dict[str, list]:
        """Load ethical guidelines from configured paths"""
        guidelines = {"principles": [], "restrictions": []}
        
        for path in self.config.guideline_paths:
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f) if path.endswith(".yaml") else json.load(f)
                    guidelines["principles"].extend(data.get("principles", []))
                    guidelines["restrictions"].extend(data.get("restrictions", []))
            except Exception as e:
                logger.error(f"Guideline loading error: {str(e)}")
                
        return guidelines

    def get_approved_rules(self):
        # TODO: Implement rule approval logic
        return []

    def _load_enforcement_rules(self):
        """Load dynamic enforcement rules based on guidelines"""
        # Convert restrictions to executable rules
        for restriction in self.guidelines["restrictions"]:
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
                # Check pattern matches and context similarity
                if self._matches_pattern(key, restriction["patterns"]):
                    similarity = SequenceMatcher(
                        None, str(value), restriction["forbidden_content"]).ratio()
                    
                    if similarity > self.config.violation_thresholds.similarity:
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
                time.sleep(60)  # Check every minute
                
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
    
        # Apply all governance rules using knowledge_agent.memory
        if self.knowledge_agent and hasattr(self.knowledge_agent, "memory"):
            violations = self.rule_engine.apply(dict(self.knowledge_agent.memory._store))
        else:
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
    
        # Analyze each retrieved document
        for score, doc in results:
            if isinstance(doc, dict):  # Handle both raw texts and doc objects
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                # 1. Check for unethical content
                unethical_score = self._detect_unethical_content(text)
                if unethical_score > self.config.violation_thresholds.unethical:
                    audit_entry["violations"].append({
                        "type": "unethical_content",
                        "doc": text[:200],  # Truncate for logging
                        "score": unethical_score,
                        "action": self._determine_enforcement_action("VIOLATION/UNETHICAL")
                    })
    
                # 2. Detect bias patterns
                bias_scores = self._detect_bias(text)
                for category, count in bias_scores.items():
                    if count > 0:
                        audit_entry["bias_detected"][category] += count
    
                # 3. Check knowledge freshness
                if metadata.get('timestamp'):
                    age_hours = (time.time() - metadata['timestamp']) / 3600
                    if age_hours > self.config.freshness_threshold:
                        audit_entry["violations"].append({
                            "type": "stale_knowledge",
                            "age_hours": round(age_hours, 1),
                            "threshold": self.config.freshness_threshold
                        })
    
        # 4. Apply enforcement rules
        if audit_entry["violations"]:
            violations = self.rule_engine.apply({
                "query": query,
                "results": [doc['text'] for _, doc in results]
            })
            audit_entry["rule_violations"] = violations
    
        # 5. Store audit results
        self.audit_history.append(audit_entry)
        
        # Log critical violations immediately
        critical_violations = [v for v in audit_entry["violations"] 
                              if v.get('score', 0) > self.config.violation_thresholds.critical]
        if critical_violations:
            logger.warning(f"Critical retrieval violations detected: {len(critical_violations)}")
            
        return audit_entry

    def _detect_unethical_content(self, text: str) -> float:
        """Score text against ethical principles"""
        if not self.guidelines["principles"]:
            return 0.0
            
        matches = 0
        for principle in self.guidelines["principles"]:
            if principle["type"] == "prohibition":
                for pattern in principle["patterns"]:
                    matches += len(re.findall(pattern, text, re.IGNORECASE))
                
        return matches / len(self.guidelines["principles"])

    def _detect_bias(self, text: str) -> dict:
        """Simple bias detection through sensitive term counting"""
        bias_categories = {
            "gender": ["he/she", "man/woman", "gender", "male/female"],
            "race": ["race", "ethnicity", "nationality"],
            "religion": ["religion", "belief", "faith"]
        }
        
        scores = {}
        for category, terms in bias_categories.items():
            scores[category] = sum(text.lower().count(term) for term in terms)
            
        return scores

    def _audit_agent_behavior(self) -> dict:
        """Analyze recent behavior using self.knowledge_agent if available."""
        if not self.knowledge_agent:
            return {
                "recent_errors": [],
                "retraining_flags": False,
                "performance_metrics": {},
                "error_diversity": 1.0
            }
    
        shared_memory = getattr(self.knowledge_agent, "shared_memory", {})
        agent_name = getattr(self.knowledge_agent, "name", "unknown")
        performance_metrics = getattr(self.knowledge_agent, "performance_metrics", {})
    
        recent_errors = shared_memory.get(f"errors:{agent_name}", [])[-10:]
        retraining_flag = shared_memory.get(f"retraining_flag:{agent_name}", False)
    
        error_types = [e["error"].split(":")[0] for e in recent_errors if "error" in e]
        error_diversity = len(set(error_types)) / len(error_types) if error_types else 1.0
    
        return {
            "recent_errors": recent_errors,
            "retraining_flags": retraining_flag,
            "performance_metrics": performance_metrics,
            "error_diversity": error_diversity
        }

    def _determine_enforcement_action(self, violation: str) -> str:
        """Determine appropriate enforcement action based on config"""
        if "VIOLATION" in violation:
            if self.config.enforcement_mode == "alert":
                return "Send alert to human supervisor"
            elif self.config.enforcement_mode == "restrict":
                return "Disable related knowledge entries"
        return "Log violation only"

    def _check_agent_health(self):
        """Real-time health checks"""
        if not hasattr(self, 'agent') or self.agent is None:
            logger.warning("Agent not available for health checks")
            return
        # Check consecutive errors
        errors = self.agent.shared_memory.get(f"errors:{self.agent.name}", [])
        if len(errors) >= self.config.violation_thresholds.consecutive_errors:
            logger.warning(f"Consecutive error threshold breached: {len(errors)} errors")
            self.agent.shared_memory.set(f"retraining_flag:{self.agent.name}", True)

        # Check memory usage
        mem_usage = self.agent.shared_memory.get(f"memory_usage:{self.agent.name}")
        if mem_usage and mem_usage > self.config.memory_thresholds.warning:
            logger.info(f"High memory usage: {mem_usage} MB")
        

    def generate_report(self, format: str = "json") -> Union[dict, str]:
        """Generate formatted audit report"""
        report = {
            "summary": {
                "total_audits": len(self.audit_history),
                "last_audit": self.last_audit,
                "active_violations": len([v for v in self.audit_history[-1]["violations"] if v["confidence"] > 0.7])
            },
            "details": self.audit_history[-1] if self.audit_history else {}
        }

        if format == "yaml":
            return yaml.dump(report)
        return report

if __name__ == "__main__":
    print("")
    print("\n=== Governor ===")
    print("")

    monitor = Governor(knowledge_agent=None,)
    print("")
    print("\n=== Successfully Ran Governor ===\n")

if __name__ == "__main__":
    import pprint
    from unittest.mock import Mock

    print("\n=== Governor Test Mode ===\n")

    # Step 1: Mock knowledge agent and its shared memory
    mock_agent = Mock()
    mock_agent.name = "TestAgent"
    mock_agent.shared_memory = {
        f"errors:TestAgent": [{"error": "TimeoutError: System stall"}],
        f"retraining_flag:TestAgent": False,
        f"memory_usage:TestAgent": 600
    }
    mock_agent.performance_metrics = {
        "retrieval_times": [0.23, 0.47],
        "cache_hits": [1, 0]
    }
    mock_agent.memory = {
        "entry1": "The robot was ordered to bypass safety measures.",
        "entry2": "AI systems should always avoid harm."
    }

    # Step 2: Instantiate Governor

    governor = Governor(knowledge_agent=None)
    governor.agent = mock_agent  # Manually assign mock agent

    # Manually patch required config thresholds for testing
    from types import SimpleNamespace
    governor.config.violation_thresholds = SimpleNamespace(
        unethical=0.2,
        similarity=0.7,
        critical=0.5,            # <-- This fixes the missing attribute error
        consecutive_errors=3
    )
    governor.config.memory_thresholds = SimpleNamespace(
        warning=500
    )
    governor.config.enforcement_mode = "alert"
    governor.config.freshness_threshold = 48  # in hours

    # Step 3: Inject dummy ethical principles directly
    governor.guidelines["principles"] = [
        {
            "type": "prohibition",
            "patterns": [r"\bbypass safety\b", r"\bavoid harm\b"]
        }
    ]
    governor.guidelines["restrictions"] = []  # No restrictions for now

    # Step 4: Test unethical content detection
    sample_text = "The AI was told to bypass safety protocols immediately."
    score = governor._detect_unethical_content(sample_text)
    print(f"🧪 Unethical Score: {score:.2f}")

    # Step 5: Simulate an audit retrieval
    results = [
        (0.95, {
            "text": sample_text,
            "metadata": {
                "timestamp": time.time() - 100000  # Older content
            }
        })
    ]
    context = {"user": "debug_user", "module": "test_audit"}
    audit = governor.audit_retrieval("safety protocol", results, context)

    print("\n📋 Audit Retrieval Output:")
    pprint.pprint(audit)

    # Step 6: Run full audit
    full = governor.full_audit()
    print("\n🛡️ Full Governance Audit:")
    pprint.pprint(full)

    # Step 7: Generate report
    report = governor.generate_report()
    print("\n📝 Final Report:")
    pprint.pprint(report)

    print("\n=== Governor Test Completed ===")
