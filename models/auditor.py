import ast
import os
import logging

class CodeAuditor:
    def __init__(self, target_path):
        self.target_path = target_path
        self.logger = logging.getLogger("SLAI.CodeAuditor")

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
                                if isinstance(node, ast.FunctionDef) and not any(isinstance(n, ast.Call) and n.func.id == node.name for n in ast.walk(tree)):
                                    issues.append((filepath, node.lineno, f"Unused function: {node.name}"))
                        except Exception as e:
                            issues.append((filepath, 0, f"Parse error: {e}"))
        return issues

    def log_issues(self, issues):
        for path, line, msg in issues:
            self.logger.warning(f"[AUDIT] {path}:{line} - {msg}")
