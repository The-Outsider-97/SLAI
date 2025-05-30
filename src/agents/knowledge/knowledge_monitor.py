
import yaml, json
import time, os
import hashlib
import threading

from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import Dict, Union, List
from types import SimpleNamespace

from src.agents.knowledge.knowledge_cache import KnowledgeCache
from src.agents.knowledge.perform_action import PerformAction
from src.agents.knowledge.governor import Governor
from src.agents.knowledge.rule_engine import RuleEngine
from logs.logger import get_logger

logger = get_logger("Knowledge Monitor")

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

class KnowledgeMonitor:
    """
    The Monitoring System integrates Academic Integrity and Knowledge Architecture Monitoring System
    """
    def __init__(self, agent=None,
                 config_section_name: str = "knowledge_monitor",
                 config_file_path: str = os.path.abspath(CONFIG_PATH)
                 ):
        self.agent = agent
        self.config = get_config_section(config_section_name, config_file_path)
        self.academic_sources = self._load_academic_sources()
        self.integrity_hashes = {}
        self.monitoring_active = False

        self.governor = Governor(knowledge_agent=agent)
        self.governor.agent = self.agent

        self.knowledge_cache = KnowledgeCache()
        self.rule_engine = RuleEngine()
        self.perform_action = PerformAction()

        if self.config.enabled:
            self._start_monitoring_thread()

    def _load_academic_sources(self) -> Dict:
        """Load academic source database from configured paths"""
        sources = {"domains": set(), "papers": [], "datasets": []}
        
        for path in self.config.academic_source_paths:
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)
                    sources["domains"].update(data.get("domains", []))
                    sources["papers"].extend(data.get("papers", []))
                    sources["datasets"].extend(data.get("datasets", []))
            except Exception as e:
                logger.error(f"Academic source loading error: {str(e)}")
                
        return sources

    def _start_monitoring_thread(self):
        """Start background monitoring of knowledge integrity"""
        def monitor():
            self.monitoring_active = True
            while self.monitoring_active:
                self.check_academic_compliance()
                self.verify_data_integrity()
                time.sleep(self.config.check_interval)
                
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def check_academic_compliance(self):
        """Validate knowledge sources against academic standards"""
        violations = []
        
        # Check source domains
        for entry in self.academic_sources["papers"]:
            if not self._is_valid_source(entry["source"]):
                violations.append({
                    "type": "invalid_domain",
                    "entry": entry["title"],
                    "source": entry["source"]
                })
            
            # Check publication freshness
            pub_date = datetime.strptime(entry["published"], "%Y-%m-%d")
            if (datetime.now() - pub_date) > timedelta(days=365*self.config.max_source_age):
                violations.append({
                    "type": "outdated_source",
                    "entry": entry["title"],
                    "age_years": (datetime.now() - pub_date).days // 365
                })
        
        # Log and handle violations
        if violations:
            logger.warning(f"Academic compliance violations detected: {len(violations)}")
            self._handle_violations(violations)

    def _is_valid_source(self, url: str) -> bool:
        """Check if URL belongs to approved academic domains"""
        domain = urlparse(url).netloc
        return any(approved in domain for approved in self.config.allowed_academic_domains)

    def verify_data_integrity(self):
        """Check hashes of critical knowledge assets"""
        if not self.config.enable_data_integrity_checks:
            return
            
        current_hashes = {
            "papers": {p["title"]: self._generate_hash(p) for p in self.academic_sources["papers"]},
            "datasets": {d["name"]: self._generate_hash(d) for d in self.academic_sources["datasets"]}
        }
        
        # Detect changes from previous hashes
        for asset_type in ["papers", "datasets"]:
            for name, new_hash in current_hashes[asset_type].items():
                old_hash = self.integrity_hashes.get(name)
                if old_hash and new_hash != old_hash:
                    logger.critical(f"Data integrity breach detected in {asset_type[:-1]}: {name}")
                    
        self.integrity_hashes = current_hashes

    def _generate_hash(self, data: Dict) -> str:
        """Generate deterministic hash for data validation"""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _handle_violations(self, violations: List[Dict]):
        """Handle compliance violations with cross-component integration"""
        action = self.config.violation_policy

        # Create violation records with temporal context
        timestamp = time.time()
        violation_records = [{
            **v,
            "timestamp": timestamp,
            "component": "KnowledgeMonitor",
            "severity": self._calculate_severity(v['type'])
        } for v in violations]

        # Store in shared memory for system-wide visibility
        self.agent.shared_memory.setdefault("violations", []).extend(violation_records)

        # Integrate with Governor's audit system
        if self.governor:
            self.governor.record_violations(violation_records)

        # Action implementations
        if action == "log":
            for v in violations:
                logger.info(f"Compliance issue: {v['type']} - {v.get('entry', '')}")
        elif action == "quarantine":
            self._quarantine_assets(violations)
        elif action == "alert":
            self._trigger_system_alert(violations)

        # Update knowledge cache
        self._invalidate_affected_knowledge(violations)

    def _calculate_severity(self, violation_type: str) -> int:
        """Determine severity based on violation type and config"""
        severity_map = {
            "invalid_domain": 3,
            "outdated_source": 2,
            "data_tampering": 4
        }
        return severity_map.get(violation_type, 1)

    def _quarantine_assets(self, violations: List[Dict]):
        """Isolate problematic assets across components"""
        quarantined = []
        
        for v in violations:
            # Remove from active sources
            if v['type'] == "invalid_domain":
                self.academic_sources["papers"] = [
                    p for p in self.academic_sources["papers"]
                    if p["source"] != v['source']
                ]
                quarantined.append(v['entry'])

            # Flag in knowledge cache
            if "entry" in v:
                key = self.knowledge_cache.hash_query(v['entry'])
                self.knowledge_cache.set(key, {"status": "quarantined"})

        logger.warning(f"Quarantined {len(quarantined)} assets")

    def _trigger_system_alert(self, violations: List[Dict]):
        """Initiate cross-component alert response"""
        # Generate comprehensive report
        alert_report = {
            "trigger": "knowledge_violation",
            "violations": violations,
            "system_status": {
                "memory_usage": self.agent.shared_memory.get("memory_usage"),
                "cache_health": len(self.knowledge_cache.cache),
                "active_rules": len(self.rule_engine.rules)
            }
        }

        # Route through Governor for enforcement
        if self.governor:
            self.governor.handle_emergency_alert(alert_report)

        # Trigger cache flush if needed
        if self.config.auto_flush_on_alert:
            self.knowledge_cache.flush_flagged_entries()

    def _invalidate_affected_knowledge(self, violations: List[Dict]):
        """Propagate invalidation through knowledge components"""
        invalid_entries = [v['entry'] for v in violations if "entry" in v]
        
        # Update knowledge cache
        for entry in invalid_entries:
            key = self.knowledge_cache.hash_query(entry)
            if cached := self.knowledge_cache.get(key):
                cached["validation_status"] = "suspect"
                self.knowledge_cache.set(key, cached)
        
        # Update rule engine context
        self.rule_engine.update_validation_context(
            invalid_entries=invalid_entries,
            trust_scores={e: 0.2 for e in invalid_entries}
        )

    def generate_academic_report(self) -> Dict:
        """Generate academic integrity report"""
        return {
            "timestamp": time.time(),
            "sources_checked": len(self.academic_sources["papers"] + self.academic_sources["datasets"]),
            "valid_domains": list(self.academic_sources["domains"]),
            "integrity_checksum": self._generate_hash(self.integrity_hashes)
        }

    def stop_monitoring(self):
        """Gracefully stop monitoring thread"""
        self.monitoring_active = False

if __name__ == "__main__":
    print("")
    print("\n=== Knowledge Monitor ===")
    print("")
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    monitor = KnowledgeMonitor(agent=mock_agent)
    print("")
    print("\n=== Successfully Monitored the Knowledge system ===\n")
