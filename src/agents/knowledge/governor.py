import yaml, json
import time
import re

from typing import Dict, Union
from difflib import SequenceMatcher
from collections import deque
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
    def __init__(self, agent=None,
                 config_section_name: str = "governor",
                 config_file_path: str = CONFIG_PATH):
        self.agent = agent
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

    def _init_rule_engine(self):
        """Initialize standalone rule engine"""
        from src.agents.knowledge.rule_engine import RuleEngine
        return RuleEngine(config_file_path=CONFIG_PATH)

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
        
        # Apply all governance rules
        violations = self.rule_engine.apply(self.agent.memory)
        for fact, confidence in violations.items():
            audit_report["violations"].append({
                "fact": fact,
                "confidence": confidence,
                "action": self._determine_enforcement_action(fact)
            })
        
        self.audit_history.append(audit_report)
        self.last_audit = time.time()
        return audit_report

    def _detect_unethical_content(self, text: str) -> float:
        """Score text against ethical principles"""
        if not self.guidelines["principles"]:
            return 0.0
            
        matches = 0
        for principle in self.guidelines["principles"]:
            if principle["type"] == "prohibition":
                matches += len(re.findall(principle["patterns"], text, re.IGNORECASE))
                
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
        """Analyze recent agent activities"""
        behavior_stats = {
            "recent_errors": self.agent.shared_memory.get(f"errors:{self.agent.name}", [])[-10:],
            "retraining_flags": self.agent.shared_memory.get(f"retraining_flag:{self.agent.name}", False),
            "performance_metrics": self.agent.performance_metrics
        }
        
        # Check for error patterns
        error_types = [e["error"].split(":")[0] for e in behavior_stats["recent_errors"]]
        behavior_stats["error_diversity"] = len(set(error_types)) / len(error_types) if error_types else 1.0
        
        return behavior_stats

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
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    monitor = Governor(agent=mock_agent)
    print("")
    print("\n=== Successfully Ran Governor ===\n")
