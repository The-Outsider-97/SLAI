import ast
import os
import logging
import subprocess
import ast, hashlib, json

from pathlib import Path

class CodeAuditor:
    def __init__(self, target_path, cache_file="logs/code_audit_cache.json"):
        self.target_path = target_path
        self.cache_file = Path(cache_file)
        self.logger = logging.getLogger("SLAI.CodeAuditor")
        self.issue_cache = self._load_cache()

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

    def run_audit(self):
        issues = []
        for root, _, files in os.walk(self.target_path):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        try:
                            tree = ast.parse(f.read(), filename=filepath)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef) and not any(
                                    isinstance(n, ast.Call) and getattr(n.func, 'id', None) == node.name
                                    for n in ast.walk(tree)
                                ):
                                    issues.append((filepath, node.lineno, f"Unused function: {node.name}"))
                        except Exception as e:
                            issues.append((filepath, 0, f"Parse error: {e}"))
        return issues

    def log_issues(self, issues):
        new_issues = []
        for path, line, msg in issues:
            h = self._hash_issue(path, line, msg)
            if h not in self.issue_cache:
                self.issue_cache.add(h)
                new_issues.append((path, line, msg))
                self.logger.warning(f"[AUDIT][NEW] {path}:{line} - {msg}")
        self._save_cache()
        return new_issues

    def run_pylint_scan(target_path="src/", log_path="logs/pylint_audit.log"):
        """Run pylint on the target path and store output in log file."""
        logger = logging.getLogger("SLAI.Pylint")
        try:
            result = subprocess.run(
                ["pylint", target_path, "--output-format=text", "--score=n"],
                capture_output=True,
                text=True,
                check=False  # Let it log warnings even if lint fails
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)
    
            logger.info(f"Pylint scan completed. Output saved to {log_path}")
        except Exception as e:
            logger.error(f"Pylint execution failed: {e}")
