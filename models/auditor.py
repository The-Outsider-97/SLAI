import ast
import os
import time
import yaml
import logging
import threading
import subprocess
import hashlib
import json
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QObject
from src.agents.evaluation_agent import EvaluationAgent
from logs.logger import get_log_queue, get_logger

logger = get_logger("SLAI.IdleAuditor")

# --- Helper: Undefined Variable Checker ---
class UndefinedVariableVisitor(ast.NodeVisitor):
    """
    Attempts to find potential usage of undefined local variables within functions.
    Note: This is a simplified check and might have false positives/negatives,
          especially with complex scopes, imports, or dynamic assignments.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.issues = []
        self.defined_vars = set()
        self.in_function_scope = False

    def visit_FunctionDef(self, node):
        # Track defined vars within this function scope
        original_defined_vars = self.defined_vars.copy()
        self.in_function_scope = True
        # Add arguments to defined vars
        self.defined_vars.update(arg.arg for arg in node.args.args)
        self.generic_visit(node) # Visit nodes inside the function
        self.defined_vars = original_defined_vars # Restore outer scope
        self.in_function_scope = False

    def visit_Assign(self, node):
        # Add assigned variables to defined set *after* visiting right side
        self.generic_visit(node) # Visit RHS first
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_vars.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)): # Handle tuple/list unpacking
                 for elt in target.elts:
                     if isinstance(elt, ast.Name):
                         self.defined_vars.add(elt.id)

    def visit_Name(self, node):
        if self.in_function_scope and isinstance(node.ctx, ast.Load):
            # Check for non-local definitions
            current_scope_vars = self.defined_vars
            in_outer_scope = any(node.id in scope for scope in self.scope_stack[:-1])
            is_import = any(node.id in self.imported_names for node in ast.walk(self.tree))

            if (not in_outer_scope and 
                not is_import and 
                not self._is_builtin_or_global(node.id)):
                self.issues.append((
                    self.filepath, node.lineno, 
                    f"Potentially undefined variable: {node.id} (Scope: {self.current_scope})"
                ))
            # Basic check: is it defined locally or seems like a builtin/global?
            # This is imperfect and could be improved (e.g., tracking globals/nonlocals)
            if node.id not in self.defined_vars and not self._is_potentially_builtin_or_global(node.id):
                 # Check if it was defined *after* this line in the current scope (simple heuristic)
                 is_defined_later = False
                 # A more robust check would analyze control flow, this is basic
                 # For simplicity, we'll just flag potential issues found here.
                 self.issues.append((self.filepath, node.lineno, f"Potential undefined variable use: {node.id}"))
        self.generic_visit(node)

    def _is_potentially_builtin_or_global(self, name):
        # Simple heuristic: common builtins or ALL_CAPS convention for constants
        # A more robust solution would involve analysing import statements and global declarations
        return name in __builtins__ or name.isupper()

# --- AuditEnhancer remains the same ---
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

        # Find the scope (function/method) for each return
        # Note: A more complex analysis might be needed for nested functions
        func_scopes = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                # Approximation: end line is the start of the next node or end of file
                # A more accurate end line requires parsing context, ast doesn't store it directly easily
                end_line = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else start_line + 100 # Heuristic
                for r_line in lines:
                    if start_line <= r_line <= end_line:
                         if node.name not in func_scopes:
                             func_scopes[node.name] = {'start': start_line, 'end': end_line, 'returns': set()}
                         func_scopes[node.name]['returns'].add(r_line)


        for node in ast.walk(tree):
             # Check if node is within a function that has a return
             if hasattr(node, 'lineno'):
                 for func_name, scope in func_scopes.items():
                     if scope['start'] <= node.lineno <= scope['end']:
                         # Is this node's line number *after* any return line within the *same scope*?
                         if any(node.lineno > r_line for r_line in scope['returns']):
                             # Avoid flagging docstrings or comments if possible (hard w/o source mapping)
                             # Avoid flagging function/class definitions themselves
                             if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Expr)): # Expr can be docstring
                                 logic_after.append((node.lineno, func_name))


        # Return unique line numbers where logic follows a return within its function scope
        # We return the line number of the offending logic
        return sorted(list(set(line for line, func in logic_after)))


# --- Updated CodeAuditor ---
class CodeAuditor(QObject):
    audit_signal = pyqtSignal(str, str)  # path, message

    # Added config for skipping checks and paths
    def __init__(self, target_path,
                 cache_file="logs/code_audit_cache.json",
                 skip_checks=None, # e.g., ["UNUSED_FUNC", "UNDEFINED_VAR"]
                 skip_paths=None, # e.g., ["*/tests/*", "temp/"]
                 pylint_rcfile=None): # Path to .pylintrc
        super().__init__()
        self.target_path = Path(target_path)
        self.cache_file = Path(cache_file)
        self.logger = get_logger("SLAI.CodeAuditor")
        self.issue_cache = self._load_cache()
        self.recent_issues = []

        # Configuration
        self.skip_checks = set(skip_checks) if skip_checks else set()
        self.skip_checks = set(skip_checks or [])
        self.skip_paths = [str(Path(p).resolve()) for p in (skip_paths or [])]
        self.pylint_rcfile = Path(pylint_rcfile) if pylint_rcfile else None

    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    # Store as dict for potential future metadata (e.g., timestamp)
                    return {h: True for h in json.load(f)}
            except Exception as e:
                self.logger.warning(f"Failed to load audit cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                # Save only the hashes (keys)
                json.dump(list(self.issue_cache.keys()), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save audit cache: {e}")

    # Use a more descriptive issue code in the hash
    def _hash_issue(self, path, line, code, msg):
        salt = os.urandom(16).hex()
        return hashlib.blake2b(
            f"{salt}|{path}|{line}|{code}|{msg}".encode(),
            digest_size=32
        ).hexdigest()

    def _should_skip(self, filepath_str):
        """Check if the filepath matches any skip patterns."""
        filepath = Path(filepath_str)
        for pattern in self.skip_paths:
            try:
                # Use glob matching for patterns like '*/tests/*'
                if filepath.match(pattern):
                    return True
            except Exception as e: # Catch potential issues with invalid patterns
                self.logger.warning(f"Error matching skip pattern '{pattern}': {e}")
        return False

    def _add_issue(self, issues, issue_code, filepath, lineno, msg):
        """Helper to add issues respecting skip_checks."""
        if issue_code not in self.skip_checks:
            issues.append({"code": issue_code, "path": filepath, "line": lineno, "msg": msg})

    def _audit_file(self, filepath):
        issues = []
        filepath_str = str(filepath)

        if self._should_skip(filepath_str):
            self.logger.debug(f"Skipping audit for path: {filepath_str}")
            return issues

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            self._add_issue(issues, "FILE_READ_ERROR", filepath_str, 0, f"Could not read file: {e}")
            return issues

        if filepath.suffix == '.py':
            try:
                tree = ast.parse(source, filename=filepath_str)

                # 1. Unused Functions (existing check)
                defined_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
                called_funcs = {
                    node.func.id for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                # Refinement: Consider functions called via getattr or from other modules (harder)
                # This check is basic and might have false positives for dynamically used functions.
                unused_funcs = defined_funcs - called_funcs
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in unused_funcs:
                        # Avoid flagging private/special methods for now unless explicitly configured
                        if not node.name.startswith('_'):
                             self._add_issue(issues, "UNUSED_FUNC", filepath_str, node.lineno, f"Potentially unused function: {node.name}")

                    # 2. Undefined Function Calls (existing check)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                         # Improve check: needs context of imports and class methods
                         # This check remains basic.
                         if node.func.id not in defined_funcs and node.func.id not in __builtins__:
                              # Further check needed: Is it an imported name? Is it a class method?
                              # Pylint is generally better at this.
                              self._add_issue(issues, "UNDEFINED_CALL", filepath_str, node.lineno, f"Potentially undefined function call: {node.func.id}")

                    # 3. Suboptimal Init Order (Basic Check)
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                             if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                                 # Check if super().__init__ is called, and if it's called early
                                 # This requires checking base classes which ast doesn't resolve easily.
                                 # Basic check: if 'super' call exists, is it the first statement (excluding docstring)?
                                 super_called = False
                                 first_stmt_index = 0
                                 if item.body and isinstance(item.body[0], ast.Expr) and isinstance(item.body[0].value, ast.Constant):
                                     first_stmt_index = 1 # Skip docstring

                                 if len(item.body) > first_stmt_index:
                                     first_real_stmt = item.body[first_stmt_index]
                                     if isinstance(first_real_stmt, ast.Expr) and \
                                        isinstance(first_real_stmt.value, ast.Call) and \
                                        isinstance(first_real_stmt.value.func, ast.Attribute) and \
                                        isinstance(first_real_stmt.value.func.value, ast.Call) and \
                                        isinstance(first_real_stmt.value.func.value.func, ast.Name) and \
                                        first_real_stmt.value.func.value.func.id == 'super' and \
                                        first_real_stmt.value.func.attr == '__init__':
                                          super_called = True
                                     elif isinstance(first_real_stmt, ast.Expr) and \
                                          isinstance(first_real_stmt.value, ast.Call) and \
                                          isinstance(first_real_stmt.value.func, ast.Attribute) and \
                                          hasattr(first_real_stmt.value.func, 'value') and \
                                          isinstance(first_real_stmt.value.func.value, ast.Name) and \
                                          first_real_stmt.value.func.value.id in node.bases: # Direct parent call Myparent.__init__(self,..)
                                          # This is more complex to verify correctly just with AST
                                          pass


                                 # This check is very basic, Pylint does it better (W0231, W0233)
                                 # if not super_called and len(node.bases) > 0:
                                 #   self._add_issue(issues, "MISSING_SUPER_INIT", filepath_str, item.lineno, f"Potential missing or late call to super().__init__ in class {node.name}")


                # 4. Logic After Return (existing check)
                enhancer = AuditEnhancer()
                logic_lines = enhancer.find_logic_after_return(tree)
                for line_no in logic_lines:
                    self._add_issue(issues, "LOGIC_AFTER_RETURN", filepath_str, line_no, "Block sequence anomaly: potentially unreachable logic after return")

                # 5. Potential Undefined Variable Use (New Check)
                var_visitor = UndefinedVariableVisitor(filepath_str)
                var_visitor.visit(tree)
                for path, line, msg in var_visitor.issues:
                     self._add_issue(issues, "UNDEFINED_VAR", path, line, msg)


            # Catch Syntax/Indentation Errors during parsing
            except (SyntaxError, IndentationError) as e:
                line = e.lineno if hasattr(e, 'lineno') else 0
                error_type = "INDENTATION_ERROR" if isinstance(e, IndentationError) else "SYNTAX_ERROR"
                self._add_issue(issues, error_type, filepath_str, line, f"Syntax error: {e.msg}")
            except Exception as e: # Catch other potential AST processing errors
                self._add_issue(issues, "AST_PARSE_ERROR", filepath_str, 0, f"Failed to parse Python file: {e}")

        elif filepath.suffix in ('.json', '.yaml', '.yml'):
            yaml_loaded = False
            try:
                if filepath.suffix == '.json':
                    json.loads(source)
                else:
                    # Ensure PyYAML is installed or handle ImportError
                    try:
                        yaml.safe_load(source)
                        yaml_loaded = True
                    except ImportError:
                         self._add_issue(issues, "YAML_IMPORT_ERROR", filepath_str, 0, "PyYAML not installed, skipping YAML checks")
                    except yaml.YAMLError as e:
                         # Try to get line number from YAMLError if available
                         mark = e.problem_mark
                         line = mark.line + 1 if mark else 0
                         self._add_issue(issues, "YAML_SYNTAX_ERROR", filepath_str, line, f"Syntax error in YAML file: {e}")

            except json.JSONDecodeError as e:
                 self._add_issue(issues, "JSON_SYNTAX_ERROR", filepath_str, e.lineno, f"Syntax error in JSON file: {e.msg}")
            except Exception as e: # Catch other errors during loading
                 self._add_issue(issues, "CONFIG_LOAD_ERROR", filepath_str, 0, f"Error loading config file: {e}")

        return issues

    def run_audit(self):
        all_issues = []
        self.logger.info(f"Starting audit on path: {self.target_path}")
        for filepath in self.target_path.rglob('*'): # Use rglob for recursive search
             if filepath.is_file() and filepath.suffix in (".py", ".json", ".yaml", ".yml"):
                 filepath_str = str(filepath)
                 # Double check skipping here as rglob might catch things os.walk misses
                 if not self._should_skip(filepath_str):
                      self.logger.debug(f"Auditing file: {filepath_str}")
                      all_issues.extend(self._audit_file(filepath))
                 else:
                     self.logger.debug(f"Skipping audit for: {filepath_str}")


        self.logger.info(f"Custom audit found {len(all_issues)} potential issues.")
        return all_issues

    def log_issues(self, issues):
        new_issues = []
        self.recent_issues.clear()
        logged_hashes = set() # Prevent logging duplicate new issues within the same run

        for issue in issues:
            path, line, code, msg = issue["path"], issue["line"], issue["code"], issue["msg"]
            h = self._hash_issue(path, line, code, msg)

            if h not in self.issue_cache and h not in logged_hashes:
                self.issue_cache[h] = True # Add to cache
                logged_hashes.add(h)       # Mark as logged in this run
                new_issues.append(issue) # Keep the dict structure

                issue_msg = f"[AUDIT][{code}] {path}:{line} - {msg}"
                self.logger.warning(issue_msg)
                self.recent_issues.append(issue_msg)
                self.audit_signal.emit(path, issue_msg) # Signal for log panel
                # get_log_queue().put(issue_msg) # Uncomment if using log queue

        if logged_hashes: # Save cache only if new issues were found/added
             self._save_cache()
        return new_issues # Return list of new issue dicts

    # --- Pylint Scan Method (with rcfile support) ---
    @staticmethod
    def run_pylint_scan(target_path=".", log_path="logs/pylint_audit.log", rcfile=None):
        logger = logging.getLogger("SLAI.Pylint")
        logger.info(f"Starting Pylint scan on: {target_path}")
        command = ["pylint", target_path, "--output-format=text", "--score=n"]
        if rcfile and Path(rcfile).exists():
            command.append(f"--rcfile={rcfile}")
            logger.info(f"Using Pylint configuration: {rcfile}")
        elif rcfile:
             logger.warning(f"Pylint rcfile specified but not found: {rcfile}")

        try:
            # Consider adding --fail-under=<score> if you want it to "fail" based on score
            result = subprocess.run(
                command + [str(target_path)],
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on Pylint non-zero exit code
                encoding='utf-8', # Explicitly set encoding
                timeout=300  # 5 minute timeout
            )

            log_dir = Path(log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"--- Pylint Output ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- Pylint Errors/Warnings ---\n")
                    f.write(result.stderr)

            if result.returncode != 0:
                 logger.warning(f"Pylint finished with exit code {result.returncode}. Check log for details: {log_path}")
            else:
                 logger.info(f"Pylint scan completed successfully. Output saved to {log_path}")

            # You could optionally parse the result.stdout here to extract specific Pylint issues
            # and integrate them into the main `issues` list if desired.
    
            # Parse output for critical issues
            critical_issues = [
                line for line in result.stdout.split('\n')
                if 'fatal' in line.lower() or 'error' in line.lower()
            ]
            
            logger.info(f"Pylint found {len(critical_issues)} critical issues")
            
        except subprocess.TimeoutExpired:
            logger.error("Pylint scan timed out after 5 minutes")

        except FileNotFoundError:
             logger.error("Pylint command not found. Make sure Pylint is installed and in your PATH.")
        except Exception as e:
             logger.error(f"Pylint execution failed: {e}")


# --- IdleAuditManager likely remains similar, but pass config to CodeAuditor ---
class IdleAuditManager:
    def __init__(self, shared_memory, agent_factory, agent, target_path=".", idle_threshold=120, auditor_config=None):
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.agent = agent
        self.target_path = target_path
        self.idle_threshold = idle_threshold
        self.last_activity_time = time.time()
        self.idle_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self._stop_flag = threading.Event()

        # Pass configuration to CodeAuditor
        auditor_config = auditor_config or {}
        # Ensure target_path is passed correctly if not in auditor_config
        if 'target_path' not in auditor_config:
             auditor_config['target_path'] = target_path
        self.auditor = CodeAuditor(**auditor_config)
        # Connect signal if needed by UI
        # self.auditor.audit_signal.connect(self.handle_audit_signal)

    def start(self):
        self.idle_thread.start()
        logger.info("Idle Audit Manager started.")

    def stop(self):
        self._stop_flag.set()
        logger.info("Idle Audit Manager stopping...")
        self.idle_thread.join(timeout=5) # Wait briefly for thread to finish
        logger.info("Idle Audit Manager stopped.")

    def register_activity(self):
        self.last_activity_time = time.time()
        logger.debug("Activity registered.")

    def _idle_monitor(self):
        logger.info("Idle monitor thread started.")
        while not self._stop_flag.wait(timeout=10): # Check stop flag every 10s
            idle_time = time.time() - self.last_activity_time
            if idle_time > self.idle_threshold:
                audit_start = time.time()
                logger.info(f"Idle time ({idle_time:.0f}s) exceeded threshold ({self.idle_threshold}s). Starting audit...")
                try:
                    # Run custom audit
                    issues = self.auditor.run_audit()
                    new_issues = self.auditor.log_issues(issues)

                    # Optionally run Pylint audit
                    if "PYLINT_SCAN" not in self.auditor.skip_checks: # Example control
                        pylint_log = "logs/idle_pylint_audit.log"
                        CodeAuditor.run_pylint_scan(
                             target_path=self.target_path,
                             log_path=pylint_log,
                             rcfile=self.auditor.pylint_rcfile
                        )
                    if new_issues: # Only log to eval agent if *new* custom issues found
                        self._log_to_evaluation_agent({
                            "log_snippets": [f"[{i['code']}] {i['path']}:{i['line']}" for i in new_issues],
                            "hazards": {"code_defects": len(new_issues)},
                            "distribution_shift": self._calc_distribution_shift(),
                            "fairness_score": 0.9 - (len(new_issues) * 0.01)
                        })

                except Exception as e:
                     logger.error(f"Error during idle audit execution: {e}", exc_info=True)

                finally:
                    self.last_activity_time = time.time() - (time.time() - audit_start)

                # Reset activity time after audit to prevent immediate re-trigger
                self.register_activity()

            else:
                 logger.debug(f"Currently idle for {idle_time:.0f}s...")
        logger.info("Idle monitor thread finished.")


    # Handle list of issue dicts
    def _log_to_evaluation_agent(self, audit_summary: dict):
        if not self.agent:
            logger.warning("Evaluation agent not available, skipping audit report.")
            return
    
        try:
            log_snippets = audit_summary.get("log_snippets", [])
            code_defects = audit_summary.get("hazards", {}).get("code_defects", 0)
    
            if not isinstance(log_snippets, list) or len(log_snippets) == 0:
                logger.warning("No meaningful log snippets, skipping evaluation log.")
                return
    
            if code_defects == 0:
                status = "up"
            else:
                status = "down"
    
            # Prepare EvaluationAgent-compliant report
            evaluation_payload = {
                "log_snippets": log_snippets[:50],  # Limit to first 50 to avoid flooding
                "hazards": {
                    "code_defects": code_defects,
                    "system_failure": 0.01 * code_defects  # simple risk scaling
                },
                "failures": code_defects,
                "status": status,
                "distribution_shift": audit_summary.get("distribution_shift", 0.05),
                "fairness_score": audit_summary.get("fairness_score", 0.90),
                "operational_time": 3600,  # Assume 1 hour unless you track it better
                "test_count": 500,          # Example: how many checks run
                "coverage": 0.92,           # Example: code coverage (can be dynamic)
                "mtbf": 1800,               # Mean time between findings (seconds)
                "avg_response_time": 0.5,   # Seconds per audit (can vary)
                "tech_debt": 0.15,          # Technical debt ratio
                "vuln_count": code_defects, # Treat defects as vuln_count for risk
            }
    
            # Reward shaping - safer defaults
            rewards_a = [0.9 - 0.01 * code_defects]
            rewards_b = [0.7 - 0.01 * code_defects]
    
            # Log through EvaluationAgent
            self.agent.log_evaluation(evaluation_payload, rewards_a=rewards_a, rewards_b=rewards_b)
    
            logger.info(f"Logged audit results: {code_defects} defects reported.")
        except Exception as e:
            logger.error(f"Failed to send audit summary to EvaluationAgent: {e}", exc_info=True)

if __name__ == "__main__":
    auditor = CodeAuditor(".")
    print("\n--- Running Custom Audit ---")
    issues = auditor.run_audit()
    auditor.log_issues(issues)
    new_issues = auditor.log_issues(issues)

    print("\nAudit complete. Issues found:")
    for issue in auditor.recent_issues:
        print(issue)

    if new_issues:
        print("\nNew issues found by custom audit:")
        for issue in new_issues:
            print(f"- [{issue['code']}] {issue['path']}:{issue['line']} - {issue['msg']}")
    else:
        print("\nNo new issues found by custom audit.")

    # Run Pylint separately if needed
    print("\n--- Running Pylint Scan ---")
    CodeAuditor.run_pylint_scan(
         target_path=config["target_path"],
         rcfile=config.get("pylint_rcfile")
    )
    print("Pylint scan complete. Check logs/pylint_audit.log")
