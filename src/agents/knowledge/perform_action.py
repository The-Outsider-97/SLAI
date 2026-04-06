import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import threading
import time
import requests

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from src.agents.knowledge.utils.config_loader import get_config_section, load_global_config
from src.agents.knowledge.utils.knowledge_errors import (GovernanceViolation, ActionExecutionError,
                                                         InvalidDocumentError, KnowledgeError,
                                                         MemoryUpdateError, Severity,)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Perform Action")
printer = PrettyPrinter

ACTION_PATTERN = re.compile(r"action:(\w+):(.+)", re.IGNORECASE)
_JSON_CONTENT_TYPE_PATTERN = re.compile(r"(?:^|/|\+)(json)$", re.IGNORECASE)
_SHELL_META_PATTERN = re.compile(r"[;&|><`$]")


class ActionDatabase:
    """SQLite-backed execution log and database action interface."""

    def __init__(self, db_path: str = "actions.db"):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        db_file = Path(self.db_path)
        if db_file.parent and str(db_file.parent) not in {"", "."}:
            db_file.parent.mkdir(parents=True, exist_ok=True)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SQL query and return structured results."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            if query.strip().lower().startswith("select"):
                return [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return [{"rows_affected": cursor.rowcount}]

    def log_action(self, action_type: str, payload: str, result: Dict[str, Any]) -> None:
        """Persist an executed action result."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO actions (action_type, payload, result)
                VALUES (?, ?, ?)
                """,
                (action_type, payload, json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()


class PerformAction:
    """Actuator module that executes validated downstream actions."""

    def __init__(
        self,
        external_api_handler: Optional[Callable[[str, str], Any]] = None,
        action_db: Optional[ActionDatabase] = None,
        knowledge_memory: Optional[KnowledgeMemory] = None,
    ):
        self.config = load_global_config()
        self.enabled = bool(self.config.get("enabled", True))
        self.action_config = get_config_section("perform_action") or {}

        self.external_api_handler = external_api_handler
        self.retry_attempts = max(int(self.action_config.get("retry_attempts", 3)), 0)
        self.retry_delay = max(float(self.action_config.get("retry_delay", 2.0)), 0.0)
        self.timeout = max(float(self.action_config.get("timeout", 30)), 0.1)
        self.max_concurrent_actions = max(int(self.action_config.get("max_concurrent_actions", 5)), 1)
        self.semaphore = threading.Semaphore(self.max_concurrent_actions)
        self._result_lock = threading.Lock()

        self.action_db = action_db or ActionDatabase(self.action_config.get("db_path", "actions.db"))
        self.knowledge_memory = knowledge_memory or KnowledgeMemory()

        self.allowed_action_types = {
            str(action_type).strip().lower()
            for action_type in self.action_config.get("allowed_action_types", ["http", "database", "shell"])
            if str(action_type).strip()
        }
        self.allowed_domains = self._normalize_allowed_domains(self.action_config.get("allowed_domains", []))
        self.allowed_shell_commands = self._normalize_command_allowlist(
            self.action_config.get("allowed_shell_commands", ["ls", "cat", "grep"])
        )
        self.allowed_database_operations = {
            str(operation).strip().upper()
            for operation in self.action_config.get(
                "allowed_database_operations",
                ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE"],
            )
            if str(operation).strip()
        }
        self.enable_sandbox = bool(self.action_config.get("enable_sandbox", True))
        self.load_action_history_on_startup = bool(
            self.action_config.get("load_action_history_on_startup", False)
        )

        if self.enabled and self.load_action_history_on_startup:
            self._load_action_history()

        printer.status(
            "INIT",
            f"Perform Action initialized with concurrency={self.max_concurrent_actions}",
            "Success",
        )

    def _normalize_allowed_domains(self, allowed_domains: List[Any]) -> set[str]:
        normalized = set()
        for value in allowed_domains:
            raw = str(value).strip().lower()
            if not raw:
                continue
            parsed = urlparse(raw if "://" in raw else f"//{raw}")
            domain = parsed.netloc or parsed.path
            domain = domain.rstrip("/")
            if domain:
                normalized.add(domain)
        return normalized

    def _normalize_command_allowlist(self, commands: List[Any]) -> set[str]:
        return {str(command).strip() for command in commands if str(command).strip()}

    def _load_action_history(self) -> None:
        """Load historical actions into memory when startup history hydration is enabled."""
        try:
            results = self.action_db.execute_query(
                "SELECT * FROM actions ORDER BY timestamp DESC LIMIT 1000"
            )
            loaded = 0
            for action in results:
                try:
                    result_payload = json.loads(action["result"]) if action.get("result") else None
                except (TypeError, json.JSONDecodeError):
                    result_payload = action.get("result")

                key = f"action_{action['id']}"
                self.knowledge_memory.update(
                    key=key,
                    value={
                        "type": action.get("action_type"),
                        "payload": action.get("payload"),
                        "result": result_payload,
                    },
                    metadata={
                        "timestamp": self._coerce_timestamp(action.get("timestamp")),
                        "source": "action_db",
                    },
                )
                loaded += 1
            logger.info("Loaded %s historical actions into memory", loaded)
        except (sqlite3.Error, MemoryUpdateError, OSError, ValueError) as exc:
            error = ActionExecutionError("startup", "load_action_history", str(exc))
            error.report()
            logger.error("Failed to load action history: %s", exc)

    def from_knowledge(self, knowledge_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a document batch with bounded concurrency and explicit result locking."""
        results: List[Dict[str, Any]] = []
        threads: List[threading.Thread] = []

        for doc in knowledge_batch or []:
            thread = threading.Thread(
                target=self._process_document,
                args=(doc, results, self._result_lock),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return results

    def _process_document(
        self,
        doc: Dict[str, Any],
        result_container: List[Dict[str, Any]],
        result_lock: threading.Lock,
    ) -> None:
        """Process one document and write results under an explicit lock."""
        try:
            actions = self._extract_actions(doc)
        except InvalidDocumentError as exc:
            exc.report()
            with result_lock:
                result_container.append({"status": "rejected", "reason": str(exc)})
            return

        for action_type, payload in actions:
            with self.semaphore:
                similar_actions = self.knowledge_memory.search_values(payload)
                if similar_actions:
                    logger.info("Found %s similar actions in memory", len(similar_actions))
                    recent_action = max(
                        similar_actions,
                        key=lambda item: item[1].get("metadata", {}).get("timestamp", 0),
                    )
                    cached_result = recent_action[1].get("value", {}).get("result")
                    if cached_result:
                        with result_lock:
                            result_container.append(cached_result)
                        continue

                result = self._execute_action(action_type, payload)
                if result is not None:
                    with result_lock:
                        result_container.append(result)

    def _extract_actions(self, doc: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract structured actions from a document payload."""
        if not isinstance(doc, dict):
            raise InvalidDocumentError(doc, "Action documents must be dictionaries")

        text = doc.get("text", "")
        if text is None:
            return []
        if not isinstance(text, str):
            raise InvalidDocumentError(doc, "Document text must be a string")

        return [(match.group(1).lower(), match.group(2).strip()) for match in ACTION_PATTERN.finditer(text)]

    def _execute_action(self, action_type: str, payload: str) -> Dict[str, Any]:
        """Validate, execute, retry, and persist one action."""
        try:
            payload = self._sanitize_payload(action_type, payload)
            self._validate_action(action_type, payload)
        except GovernanceViolation as exc:
            exc.report()
            logger.warning("Rejected action %s: %s", action_type, exc)
            return {
                "action": action_type,
                "status": "rejected",
                "error": str(exc),
                "attempts": 0,
            }

        identical_actions = self.knowledge_memory.search_values(payload)
        for _, action_data in identical_actions:
            action_value = action_data.get("value", {})
            if (
                action_value.get("type") == action_type
                and action_value.get("payload") == payload
                and action_value.get("result")
            ):
                logger.info("Using identical action result from memory")
                return action_value["result"]

        last_error: Optional[Exception] = None
        for attempt in range(self.retry_attempts + 1):
            try:
                start_time = time.monotonic()
                result = self._route_action(action_type, payload)
                duration = time.monotonic() - start_time
                action_result = {
                    "action": action_type,
                    "status": "success",
                    "result": result,
                    "attempts": attempt + 1,
                    "duration": duration,
                }
                self._store_action_result(action_type, payload, action_result)
                return action_result
            except GovernanceViolation as exc:
                exc.report()
                logger.warning("Rejected action %s during execution: %s", action_type, exc)
                return {
                    "action": action_type,
                    "status": "rejected",
                    "error": str(exc),
                    "attempts": attempt + 1,
                }
            except Exception as exc:  # execution failures should flow through retry budget
                last_error = exc
                if isinstance(exc, KnowledgeError):
                    exc.report()
                else:
                    ActionExecutionError(action_type, payload, str(exc)).report()

                if attempt >= self.retry_attempts:
                    error_result = {
                        "action": action_type,
                        "status": "failed",
                        "error": str(exc),
                        "attempts": attempt + 1,
                    }
                    self._store_action_result(action_type, payload, error_result)
                    return error_result

                time.sleep(self.retry_delay)

        # Defensive fallback; the loop above should always return.
        error_result = {
            "action": action_type,
            "status": "failed",
            "error": str(last_error) if last_error else "Unknown action failure",
            "attempts": self.retry_attempts + 1,
        }
        self._store_action_result(action_type, payload, error_result)
        return error_result

    def _store_action_result(self, action_type: str, payload: str, result: Dict[str, Any]) -> None:
        """Persist action results into memory and the action database."""
        action_hash = hashlib.md5(f"{action_type}_{payload}".encode("utf-8")).hexdigest()
        action_key = f"action_{action_hash}"

        try:
            self.knowledge_memory.update(
                key=action_key,
                value={
                    "type": action_type,
                    "payload": payload,
                    "result": result,
                },
                metadata={
                    "timestamp": time.time(),
                    "source": "performed",
                },
            )
        except Exception as exc:
            error = MemoryUpdateError(action_key, result, str(exc))
            error.report()
            logger.error("Failed to store action result in memory: %s", exc)

        try:
            self.action_db.log_action(action_type, payload, result)
        except sqlite3.Error as exc:
            error = ActionExecutionError(action_type, payload, f"Failed to log action result: {exc}")
            error.report()
            logger.error("Failed to persist action result: %s", exc)

    def _validate_action(self, action_type: str, payload: str) -> None:
        """Validate action policy and payload before execution."""
        if action_type not in self.allowed_action_types:
            raise GovernanceViolation(
                "perform_action_type_allowlist",
                {"action_type": action_type, "allowed_types": sorted(self.allowed_action_types)},
                payload,
            )

        if action_type == "http":
            self._parse_http_payload(payload)
            return

        if action_type == "database":
            query_type = self._get_sql_operation(payload)
            if self.enable_sandbox and query_type not in self.allowed_database_operations:
                raise GovernanceViolation(
                    "perform_action_database_allowlist",
                    {
                        "operation": query_type,
                        "allowed_operations": sorted(self.allowed_database_operations),
                    },
                    payload,
                )
            return

        if action_type == "shell":
            self._build_shell_argv(payload)
            return

        raise GovernanceViolation(
            "perform_action_unknown_type",
            {"action_type": action_type},
            payload,
        )

    def _route_action(self, action_type: str, payload: str) -> Any:
        """Route actions to their execution backend."""
        if self.external_api_handler is not None:
            try:
                return self.external_api_handler(action_type, payload)
            except Exception as exc:
                raise ActionExecutionError(action_type, payload, str(exc)) from exc

        if action_type == "http":
            return self._execute_http(payload)
        if action_type == "database":
            return self._execute_db_query(payload)
        if action_type == "shell":
            return self._execute_shell(payload)

        raise ActionExecutionError(action_type, payload, f"Unhandled action type: {action_type}")

    def _sanitize_payload(self, action_type: str, payload: str) -> str:
        """Perform minimal normalization without mutating action intent."""
        sanitized = (payload or "").strip()
        if action_type == "database" and self.enable_sandbox:
            sanitized = re.sub(r"--.*?$", "", sanitized, flags=re.MULTILINE).strip()
            sanitized = sanitized.rstrip(";").strip()
        return sanitized

    def _parse_http_payload(self, payload: str) -> Tuple[str, str, Optional[Dict[str, Any]], Dict[str, str]]:
        parts = payload.split(maxsplit=1)
        if len(parts) != 2:
            raise ActionExecutionError("http", payload, "Invalid HTTP payload format")

        method = parts[0].upper().strip()
        if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
            raise ActionExecutionError("http", payload, f"Unsupported HTTP method: {method}")

        remainder = parts[1].strip()
        match = re.match(r"^(\S+)(?:\s+({.*))?$", remainder, flags=re.DOTALL)
        if not match:
            raise ActionExecutionError("http", payload, "Missing URL in HTTP payload")

        url = match.group(1).strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ActionExecutionError("http", payload, "Invalid HTTP URL")

        if self.allowed_domains and not self._domain_is_allowed(parsed.netloc.lower()):
            raise GovernanceViolation(
                "perform_action_domain_allowlist",
                {"domain": parsed.netloc.lower(), "allowed_domains": sorted(self.allowed_domains)},
                payload,
            )

        headers: Dict[str, str] = {}
        body: Optional[Dict[str, Any]] = None
        if match.group(2):
            try:
                body = json.loads(match.group(2))
            except json.JSONDecodeError as exc:
                raise ActionExecutionError("http", payload, f"Invalid JSON request body: {exc}") from exc
            headers["Content-Type"] = "application/json"

        return method, url, body, headers

    def _domain_is_allowed(self, domain: str) -> bool:
        return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in self.allowed_domains)

    def _execute_http(self, payload: str) -> Dict[str, Any]:
        method, url, body, headers = self._parse_http_payload(payload)
        try:
            response = requests.request(
                method=method,
                url=url,
                json=body,
                headers=headers or None,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ActionExecutionError("http", payload, str(exc)) from exc

        content_type = (response.headers.get("Content-Type") or "").strip().lower()
        json_payload = None
        if self._is_json_content_type(content_type):
            try:
                json_payload = response.json()
            except ValueError as exc:
                raise ActionExecutionError("http", payload, f"Invalid JSON response body: {exc}") from exc

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": response.text if response.text else None,
            "json": json_payload,
        }

    def _get_sql_operation(self, query: str) -> str:
        first_token = (query or "").strip().split(maxsplit=1)
        if not first_token:
            raise ActionExecutionError("database", query, "Empty database query")
        return first_token[0].upper()

    def _execute_db_query(self, query: str) -> Dict[str, Any]:
        query_type = self._get_sql_operation(query)
        try:
            results = self.action_db.execute_query(query)
        except sqlite3.Error as exc:
            raise ActionExecutionError("database", query, str(exc)) from exc

        return {
            "type": query_type,
            "results": results,
            "result_count": len(results),
        }

    def _build_shell_argv(self, command: str) -> List[str]:
        if not isinstance(command, str) or not command.strip():
            raise ActionExecutionError("shell", str(command), "Empty shell command")

        if self.enable_sandbox and _SHELL_META_PATTERN.search(command):
            raise GovernanceViolation(
                "perform_action_shell_metacharacters",
                {"command": command},
                command,
            )

        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            raise ActionExecutionError("shell", command, f"Invalid shell syntax: {exc}") from exc

        if not argv:
            raise ActionExecutionError("shell", command, "No executable provided")

        executable = argv[0]
        if executable not in self.allowed_shell_commands:
            raise GovernanceViolation(
                "perform_action_shell_allowlist",
                {"command": executable, "allowed_commands": sorted(self.allowed_shell_commands)},
                command,
            )

        return argv

    def _execute_shell(self, command: str) -> Dict[str, Any]:
        argv = self._build_shell_argv(command)
        try:
            result = subprocess.run(
                argv,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ActionExecutionError("shell", command, "Command timed out") from exc
        except OSError as exc:
            raise ActionExecutionError("shell", command, str(exc)) from exc

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "argv": argv,
        }

    def _is_json_content_type(self, content_type: str) -> bool:
        media_type = content_type.split(";", 1)[0].strip().lower()
        return media_type == "application/json" or media_type.endswith("+json")

    def _coerce_timestamp(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if not value:
            return time.time()
        if isinstance(value, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return time.mktime(time.strptime(value, fmt))
                except ValueError:
                    continue
        return time.time()


if __name__ == "__main__":  # pragma: no cover
    print("\n=== Running Perform Action ===\n")
    printer.status("Init", "Perform Action initialized", "success")

    db = ActionDatabase("test_actions.db")
    db.execute_query(
        """
        CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
        """
    )
    db.execute_query("INSERT INTO test_data (name, value) VALUES ('test1', 100)")
    db.execute_query("INSERT INTO test_data (name, value) VALUES ('test2', 200)")

    # Override allowed domains for test (allow all)
    action = PerformAction(action_db=db)
    action.allowed_domains = set()  # allow any domain during test

    print("\nTesting HTTP action:")
    http_result = action._execute_http("GET https://jsonplaceholder.typicode.com/todos/1")
    printer.pretty("HTTP Result:", http_result, "info")

    print("\nTesting DB action:")
    db_result = action._execute_db_query("SELECT * FROM test_data")
    printer.pretty("DB Result:", db_result, "info")

    print("\nTesting Shell action:")
    # Use a cross-platform safe command
    if os.name == "nt":
        shell_cmd = "dir"
    else:
        shell_cmd = "ls -l"
    try:
        shell_result = action._execute_shell(shell_cmd)
        printer.pretty("Shell Result:", shell_result, "info")
    except Exception as exc:
        printer.pretty("Shell Result:", {"error": str(exc)}, "error")

    print("\nTesting full action pipeline:")
    doc = {"text": "action:http:GET https://jsonplaceholder.typicode.com/todos/2"}
    results = action.from_knowledge([doc])
    printer.pretty("Action Result:", results, "success")
    printer.status("Details", action, "info")
    print("\n=== Successfully ran Perform Action ===\n")