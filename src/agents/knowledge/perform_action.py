
import os
import re
import time
import hashlib
import sqlite3
import requests
import threading
import subprocess
import yaml, json

from urllib.parse import urlparse
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Tuple

from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Perform Action")
printer = PrettyPrinter

ACTION_PATTERN = re.compile(r"action:(\w+):(.+)", re.IGNORECASE)

class ActionDatabase:
    """Simple database interface for action execution"""
    def __init__(self, db_path: str = "actions.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SQL query and return results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                if query.strip().lower().startswith("select"):
                    return [dict(row) for row in cursor.fetchall()]
                conn.commit()
                return [{"rows_affected": cursor.rowcount}]
            except sqlite3.Error as e:
                return [{"error": str(e)}]
    
    def log_action(self, action_type: str, payload: str, result: str):
        """Log an executed action"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO actions (action_type, payload, result)
                VALUES (?, ?, ?)
            """, (action_type, payload, json.dumps(result)))
            conn.commit()

class PerformAction:
    """
    Actuator module (Hands). Executes downstream actions based on retrieved or inferred knowledge.
    """

    def __init__(self, external_api_handler: Optional[callable] = None):
        """
        Args:
            external_api_handler (callable, optional): Function or object that executes external system commands.
        """
        self.config = load_global_config()
        self.action_config = get_config_section('perform_action')
        self.sector_rules = defaultdict(list)
        self.external_api_handler = external_api_handler
        self.semaphore = threading.Semaphore(self.action_config.get('max_concurrent_actions'))

        self.active_actions = 0
        self.action_db = ActionDatabase(self.action_config.get('db_path', 'actions.db'))

        self.knowledge_memory = KnowledgeMemory()

        printer.status("INIT", f"Perform Action Initialized with: {self.active_actions} Active Actions", "Success")

    def _load_action_history(self):
        """Load historical actions into memory"""
        try:
            results = self.action_db.execute_query("SELECT * FROM actions ORDER BY timestamp DESC LIMIT 1000")
            for action in results:
                key = f"action_{action['id']}"
                self.knowledge_memory.update(
                    key=key,
                    value={
                        "type": action["action_type"],
                        "payload": action["payload"],
                        "result": json.loads(action["result"]) if action["result"] else None
                    },
                    metadata={
                        "timestamp": action["timestamp"],
                        "source": "action_db"
                    }
                )
            logger.info(f"Loaded {len(results)} historical actions into memory")
        except Exception as e:
            logger.error(f"Failed to load action history: {str(e)}")

    def from_knowledge(self, knowledge_batch: List[Dict[str, Any]]) -> List[Dict]:
        """Process batch with enhanced parallel execution tracking"""
        results = []
        threads = []
        
        for doc in knowledge_batch:
            thread = threading.Thread(
                target=self._process_document,
                args=(doc, results))
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
                    # Check memory for similar actions
                    similar_actions = self.knowledge_memory.search_values(payload)
                    if similar_actions:
                        logger.info(f"Found {len(similar_actions)} similar actions in memory")
                        # Use the most recent similar action result if possible
                        recent_action = max(similar_actions, 
                                           key=lambda x: x[1]["metadata"].get("timestamp", 0))
                        if recent_action[1]["value"].get("result"):
                            result_container.append(recent_action[1]["value"])
                            continue
                    
                    # Execute new action if no valid similar actions found
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

        # Check if identical action exists in memory
        identical_actions = self.knowledge_memory.search_values(payload)
        if identical_actions:
            for _, action_data in identical_actions:
                action_value = action_data["value"]
                if (action_value["type"] == action_type and 
                    action_value["payload"] == payload and 
                    action_value.get("result")):
                    logger.info("Using identical action from memory")
                    return action_value["result"]

        for attempt in range(self.action_config.get('retry_attempts', 3) + 1):
            try:
                start_time = time.time()
                
                if self.action_config.get('enable_sandbox', True):
                    payload = self._sanitize_payload(action_type, payload)

                result = self._route_action(action_type, payload)
                duration = time.time() - start_time
                
                if duration > self.action_config.get('timeout', 30):
                    raise TimeoutError("Action exceeded timeout")

                # Store result in memory and database
                action_result = {
                    "action": action_type,
                    "status": "success",
                    "result": result,
                    "attempts": attempt + 1,
                    "duration": duration
                }
                self._store_action_result(action_type, payload, action_result)
                
                return action_result
                
            except Exception as e:
                if attempt == self.action_config.get('retry_attempts', 3):
                    error_result = {
                        "action": action_type,
                        "status": "failed",
                        "error": str(e),
                        "attempts": attempt + 1
                    }
                    self._store_action_result(action_type, payload, error_result)
                    return error_result
                time.sleep(self.action_config.get('retry_delay', 2))

    def _store_action_result(self, action_type: str, payload: str, result: dict):
        """Store action result in memory and database"""
        # Generate unique key based on action content
        action_hash = hashlib.md5(f"{action_type}_{payload}".encode()).hexdigest()
        
        # Store in knowledge memory
        self.knowledge_memory.update(
            key=f"action_{action_hash}",
            value={
                "type": action_type,
                "payload": payload,
                "result": result
            },
            metadata={
                "timestamp": time.time(),
                "source": "performed"
            }
        )
        
        # Store in action database
        self.action_db.log_action(action_type, payload, result)

    def _validate_action(self, action_type: str, payload: str) -> bool:
        """Multi-layer validation"""
        # Type check
        allowed_types = self.action_config.get('allowed_action_types', ["http", "database", "shell"])
        if action_type not in allowed_types:
            logger.warning(f"Blocked restricted action type: {action_type}")
            return False
            
        # Domain check for HTTP
        if action_type == "http":
            allowed_domains = self.action_config.get('allowed_domains', [])
            try:
                domain = urlparse(payload.split()[0]).netloc
                if domain and not any(domain.endswith(d) for d in allowed_domains):
                    logger.warning(f"Blocked restricted domain: {domain}")
                    return False
            except:
                return False
                
        # Shell command validation
        if action_type == "shell" and self.action_config.get('enable_sandbox', True):
            blocked_commands = ["rm", "sudo", "chmod", ">", "|", "&", ";"]
            if any(cmd in payload for cmd in blocked_commands):
                logger.warning(f"Blocked dangerous shell command: {payload}")
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
        # Remove potentially dangerous characters
        sanitized = payload
        
        if action_type == "shell":
            # Remove command chaining characters
            sanitized = re.sub(r'[;&|>]', '', payload)
            # Remove potentially dangerous commands
            sanitized = re.sub(r'\b(rm|sudo|chmod)\b', '', sanitized, flags=re.IGNORECASE)
        
        elif action_type == "database":
            # Remove comments and semicolons
            sanitized = re.sub(r'--.*?$|;', '', payload, flags=re.MULTILINE)
            
        elif action_type == "http":
            # Sanitize URL parameters
            sanitized = re.sub(r'[<>"\'%]', '', payload)
            
        return sanitized.strip()

    def _execute_http(self, payload: str) -> Dict:
        """Execute HTTP requests using requests library"""
        try:
            # Parse method and URL
            parts = payload.split(maxsplit=1)
            if len(parts) < 1:
                raise ValueError("Invalid HTTP payload format")
                
            method = parts[0].upper()
            url = parts[1] if len(parts) > 1 else ""
            
            # Handle headers and body if present
            headers = {}
            body = None
            if "{" in url and "}" in url:
                url, json_str = url.split("{", 1)
                json_str = "{" + json_str
                try:
                    body = json.loads(json_str)
                    headers['Content-Type'] = 'application/json'
                except json.JSONDecodeError:
                    # Fallback to regular URL if JSON is invalid
                    pass
            
            logger.info(f"HTTP {method} to {url}")
            
            # Execute request
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, json=body, headers=headers)
            elif method == "PUT":
                response = requests.put(url, json=body, headers=headers)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text if response.text else None,
                "json": response.json() if response.headers.get('Content-Type') == 'application/json' else None
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"HTTP error: {str(e)}"}

    def _execute_db_query(self, query: str) -> Dict:
        """Execute database queries"""
        try:
            # Parse query type (SELECT/INSERT/UPDATE/DELETE)
            query_type = query.strip().split()[0].upper()
            
            # Execute using ActionDatabase
            results = self.action_db.execute_query(query)
            
            return {
                "type": query_type,
                "results": results,
                "result_count": len(results)
            }
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}

    def _execute_shell(self, command: str) -> Dict:
        """Execute shell commands in a sandboxed environment"""
        try:
            # Validate command against allow-list if configured
            allowed_commands = self.action_config.get('allowed_shell_commands', ['ls', 'cat', 'grep'])
            if allowed_commands:
                base_cmd = command.split()[0]
                if base_cmd not in allowed_commands:
                    raise ValueError(f"Command not allowed: {base_cmd}")
            
            # Execute in a safe environment
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.action_config.get('timeout', 30)
            )
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out"}
        except Exception as e:
            return {"error": f"Shell error: {str(e)}"}


if __name__ == "__main__":
    print("\n=== Running Perform Action ===\n")
    printer.status("Init", "Perform Action initialized", "success")

    # Create test database
    db = ActionDatabase("test_actions.db")
    db.execute_query("""
        CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
    """)
    db.execute_query("INSERT INTO test_data (name, value) VALUES ('test1', 100)")
    db.execute_query("INSERT INTO test_data (name, value) VALUES ('test2', 200)")

    # external_api_handler
    action = PerformAction()

    # Test HTTP action
    print("\nTesting HTTP action:")
    http_result = action._execute_http("GET https://jsonplaceholder.typicode.com/todos/1")
    printer.pretty("HTTP Result:", http_result, "info")
    
    # Test DB action
    print("\nTesting DB action:")
    db_result = action._execute_db_query("SELECT * FROM test_data")
    printer.pretty("DB Result:", db_result, "info")
    
    # Test Shell action
    print("\nTesting Shell action:")
    shell_result = action._execute_shell("ls -l")
    printer.pretty("Shell Result:", shell_result, "info")
    
    # Test full action execution
    print("\nTesting full action pipeline:")
    doc = {"text": "action:http:GET https://jsonplaceholder.typicode.com/todos/2"}
    results = action.from_knowledge([doc])
    printer.pretty("Action Result:", results, "success")
    printer.status("Details", action, "info")
    print("\n=== Succesfully ran Perform Action ===\n")
