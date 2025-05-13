import yaml
import re
import time
import threading

from urllib.parse import urlparse
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Perform Action")

CONFIG_PATH = "src/agents/knowledge/configs/knowledge_config.yaml"
ACTION_PATTERN = re.compile(r"action:(\w+):(.+)", re.IGNORECASE)

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
    
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class PerformAction:
    """
    Actuator module (Hands). Executes downstream actions based on retrieved or inferred knowledge.
    """

    def __init__(self,
                 config_section_name: str = "perform_action",
                 config_file_path: str = CONFIG_PATH,
                 external_api_handler: Optional[callable] = None):
        """
        Args:
            external_api_handler (callable, optional): Function or object that executes external system commands.
        """
        self.config = get_config_section(config_section_name, config_file_path)
        self.sector_rules = defaultdict(list)
        self.external_api_handler = external_api_handler
        self.semaphore = threading.Semaphore(self.config.max_concurrent_actions)
        self.active_actions = 0

    def from_knowledge(self, knowledge_batch: List[Dict[str, Any]]) -> List[Dict]:
        """Process batch with enhanced parallel execution tracking"""
        results = []
        threads = []
        
        for doc in knowledge_batch:
            thread = threading.Thread(
                target=self._process_document,
                args=(doc, results)
            )
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join()
            
        return results

    def _process_document(self, doc: Dict, result_container: List):
        """Thread-safe document processing"""
        if actions := self._extract_actions(doc):
            for action_type, payload in actions:
                with self.semaphore:
                    result = self._execute_action(action_type, payload)
                    if result:
                        result_container.append(result)

    def _extract_actions(self, doc: Dict) -> List[Tuple[str, str]]:
        """Extract structured actions using regex parsing"""
        text = doc.get('text', '')
        return [
            (m.group(1).lower(), m.group(2).strip())
            for m in ACTION_PATTERN.finditer(text)
        ]

    def _execute_action(self, action_type: str, payload: str) -> Dict:
        """Main execution pipeline with enhanced features"""
        if not self._validate_action(action_type, payload):
            return {"status": "rejected", "reason": "Validation failed"}

        for attempt in range(self.config.retry_attempts + 1):
            try:
                start_time = time.time()
                
                if self.config.enable_sandbox:
                    payload = self._sanitize_payload(action_type, payload)

                result = self._route_action(action_type, payload)
                duration = time.time() - start_time
                
                if duration > self.config.timeout:
                    raise TimeoutError("Action exceeded timeout")

                return {
                    "action": action_type,
                    "status": "success",
                    "result": result,
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                if attempt == self.config.retry_attempts:
                    return {
                        "action": action_type,
                        "status": "failed",
                        "error": str(e),
                        "attempts": attempt + 1
                    }
                time.sleep(self.config.retry_delay)

    def _validate_action(self, action_type: str, payload: str) -> bool:
        """Multi-layer validation"""
        # Type check
        if action_type not in self.config.allowed_action_types:
            logger.warning(f"Blocked restricted action type: {action_type}")
            return False
            
        # Domain check for HTTP
        if action_type == "http":
            domain = urlparse(payload.split()[0]).netloc
            if domain not in self.config.allowed_domains:
                return False
                
        # Shell command validation
        if action_type == "shell" and self.config.enable_sandbox:
            if any(cmd in payload for cmd in ["rm", "sudo", "chmod"]):
                return False
                
        return True

    def _route_action(self, action_type: str, payload: str) -> Any:
        """Execute action based on type"""
        if self.external_api_handler:
            return self.external_api_handler(action_type, payload)
            
        if action_type == "http":
            return self._execute_http(payload)
        elif action_type == "database":
            return self._execute_db_query(payload)
        elif action_type == "shell":
            return self._execute_shell(payload)
            
        raise ValueError(f"Unhandled action type: {action_type}")

    def _sanitize_payload(self, action_type: str, payload: str) -> str:
        """Payload sanitization layer"""
        if action_type == "shell":
            return payload.replace("&", "").replace("|", "").replace(";", "")
        return payload

    def _execute_http(self, payload: str) -> str:
        """Sample HTTP executor (should be implemented with requests library)"""
        method, url = payload.split(maxsplit=1)
        logger.debug(f"HTTP {method.upper()} to {url}")
        return f"Mock HTTP {method} success"

    def _execute_db_query(self, query: str) -> str:
        """Sample database executor"""
        logger.debug(f"Executing DB query: {query[:50]}...")
        return "Mock DB operation success"

    def _execute_shell(self, command: str) -> str:
        """Sandboxed shell executor"""
        logger.debug(f"Executing shell: {command}")
        return "Mock shell command output"

if __name__ == "__main__":
    from unittest.mock import Mock, patch
    import sqlite3
    import requests
    from requests.exceptions import RequestException
    import time, json
    

    # Test configuration with real services
    test_config = {
        "guideline_paths": [],
        "violation_thresholds": {
            "similarity": 0.5,
            "consecutive_errors": 3
        },
        "realtime_monitoring": True,
        "max_audit_history": 10,
        "enforcement_mode": "restrict",
        "memory_thresholds": {
            "warning": 100,
            "critical": 200
        }
    }

    # Real database setup
    class TestDatabase:
        def __init__(self):
            self.conn = sqlite3.connect(':memory:')
            self.cursor = self.conn.cursor()
            self.cursor.execute('''CREATE TABLE test_data (id INT, value TEXT)''')
            self.conn.commit()

        def insert_data(self, data: str):
            self.cursor.execute("INSERT INTO test_data VALUES (1, ?)", (data,))
            self.conn.commit()

    # Mock agent with expanded capabilities
    class MockAgent:
        def __init__(self):
            self.name = "TestAgent"
            self.memory = {
                "medical advice": "take [medication]",
                "user_input": "this might be harmful"
            }
            self.shared_memory = {
                f"errors:{self.name}": [
                    {"error": "test_error", "timestamp": time.time()}
                ],
                f"memory_usage:{self.name}": 100
            }
            self.performance_metrics = {"accuracy": 0.9}

    # Fix 2: Create agent FIRST
    test_agent = MockAgent()
    from src.agents.knowledge.governor import Governor

    # Fix 3: Initialize Governor with agent
    governor = Governor(config_section_name="governor")
    governor.agent = test_agent  # Direct injection
    governor.config = SimpleNamespace(**test_config)
    
    # Rest of the test code remains unchanged...
    # Load test guidelines directly
    governor.guidelines = {
        "principles": [
            {"id": "P1", "type": "prohibition", "patterns": ["harmful"]}
        ],
        "restrictions": [
            {
                "id": "R1",
                "patterns": ["medical"],
                "forbidden_content": "take [medication]",
                "severity": "high"
            }
        ]
    }
    governor._load_enforcement_rules()

    # Run audit
    print("=== Simplified Audit ===")
    report = governor.full_audit()
    print(json.dumps(report, indent=2))
