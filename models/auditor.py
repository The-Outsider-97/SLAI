import ast
import os
import time
import logging
import threading
import subprocess
import hashlib
import json
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QObject
from src.agents.evaluation_agent import EvaluationAgent
from logs.logger import get_log_queue

logger = logging.getLogger("SLAI.IdleAuditor")

class AuditEnhancer(ast.NodeVisitor):
    """Detects returns followed by subsequent logic, flagging block-sequence anomalies."""
    def __init__(self):
        self.return_lines = set()

    def visit_Return(self, node):
        self.return_lines.add(node.lineno)
        self.generic_visit(node)

    def find_logic_after_return(self, tree):
        self.visit(tree)
        lines = sorted(self.return_lines)
        logic_after = []

        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and any(node.lineno > r for r in lines):
                logic_after.append(node.lineno)

        return sorted(set(logic_after))

class CodeAuditor(QObject):
    audit_signal = pyqtSignal(str, str)  # path, message

    def __init__(self, target_path, cache_file="logs/code_audit_cache.json"):
        super().__init__()
        self.target_path = target_path
        self.cache_file = Path(cache_file)
        self.logger = logging.getLogger("SLAI.CodeAuditor")
        self.issue_cache = self._load_cache()
        self.recent_issues = []  # For real-time display

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(list(self.issue_cache), f)

    def _hash_issue(self, path, line, msg):
        return hashlib.sha256(f"{path}:{line}:{msg}".encode()).hexdigest()

    def _is_builtin(self, name):
        return name in {...}  # auto-filled with safe built-in and stdlib names

    def _audit_file(self, filepath):
        issues = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            if filepath.endswith('.py'):
                tree = ast.parse(source, filename=filepath)
                defined_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
                called_funcs = {
                    node.func.id for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }

                unused_funcs = defined_funcs - called_funcs
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in unused_funcs:
                        issues.append((filepath, node.lineno, f"Unused function: {node.name}"))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if (node.func.id not in defined_funcs and not self._is_builtin(node.func.id)):
                            issues.append((filepath, node.lineno, f"Undefined function call: {node.func.id}"))

                ret_lines = AuditEnhancer().find_logic_after_return(tree)
                for ret_line in ret_lines:
                    issues.append((filepath, ret_line + 1, "Block sequence anomaly: logic after return"))
            elif filepath.endswith(('.json', '.yaml', '.yml')):
                try:
                    if filepath.endswith('.json'):
                        json.loads(source)
                    else:
                        import yaml
                        yaml.safe_load(source)
                except Exception as e:
                    issues.append((filepath, 0, f"Syntax error in config file: {str(e)}"))

        except Exception as e:
            issues.append((filepath, 0, f"Parse error: {e}"))
        return issues

    def run_audit(self):
        issues = []
        for root, _, files in os.walk(self.target_path):
            for file in files:
                if file.endswith((".py", ".json", ".yaml", ".yml")):
                    filepath = os.path.join(root, file)
                    issues.extend(self._audit_file(filepath))
        return issues

    def log_issues(self, issues):
        new_issues = []
        self.recent_issues.clear()
        for path, line, msg in issues:
            h = self._hash_issue(path, line, msg)
            if h not in self.issue_cache:
                self.issue_cache.add(h)
                new_issues.append((path, line, msg))
                issue_msg = f"[AUDIT][NEW] {path}:{line} - {msg}"
                self.logger.warning(issue_msg)
                self.recent_issues.append(issue_msg)
                self.audit_signal.emit(path, issue_msg)  # Signal for log panel
                get_log_queue().put(issue_msg)
        self._save_cache()
        return new_issues

    @staticmethod
    def run_pylint_scan(target_path="src", log_path="logs/pylint_audit.log"):
        logger = logging.getLogger("SLAI.Pylint")
        try:
            result = subprocess.run(
                ["pylint", target_path, "--output-format=text", "--score=n"],
                capture_output=True,
                text=True,
                check=False
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
            logger.info(f"Pylint scan completed. Output saved to {log_path}")
        except Exception as e:
            logger.error(f"Pylint execution failed: {e}")

class IdleAuditManager:
    def __init__(self, shared_memory, agent_factory, agent, target_path="src", idle_threshold=120):
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.agent = agent
        self.target_path = target_path
        self.idle_threshold = idle_threshold
        self.last_activity_time = time.time()
        self.idle_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self._stop_flag = threading.Event()
        self.auditor = CodeAuditor(target_path=target_path)

    def start(self):
        self.idle_thread.start()

    def stop(self):
        self._stop_flag.set()

    def register_activity(self):
        self.last_activity_time = time.time()

    def _idle_monitor(self):
        while not self._stop_flag.is_set():
            time.sleep(10)
            if time.time() - self.last_activity_time > self.idle_threshold:
                logger.info("Idle detected. Starting audit...")
                issues = self.auditor.run_audit()
                new_issues = self.auditor.log_issues(issues)
                if new_issues:
                    self._log_to_evaluation_agent(new_issues)

    def _log_to_evaluation_agent(self, issues):
        try:
            results = {
                "log_snippets": [f"{i[0]}:{i[1]} - {i[2]}" for i in issues],
                "hazards": {"code_defects": len(issues)},
                "failures": len(issues),
                "status": "down" if len(issues) > 0 else "up",
                "distribution_shift": 0.0,
                "fairness_score": 0.9
            }
            self.agent.log_evaluation(results, rewards_a=[0.9], rewards_b=[0.7])
        except Exception as e:
            logger.error(f"Failed to log audit results to EvaluationAgent: {e}")

if __name__ == "__main__":
    auditor = CodeAuditor("src")
    issues = auditor.run_audit()
    auditor.log_issues(issues)

    print("\nAudit complete. Issues found:")
    for issue in auditor.recent_issues:
        print(issue)
