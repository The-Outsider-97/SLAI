
import os
import ast
import astor
import networkx as nx

from typing import Dict, List, Optional, Any, Set, Tuple

from src.agents.evaluators.utils.evaluators_calculations import EvaluatorsCalculations
from logs.logger import get_logger

logger = get_logger("Static Analyzer")

class StaticAnalyzer:  
    def __init__(self, codebase_path: str):  
        self.ast_analyzer = ASTAnalyzer(codebase_path)  
        self.symbolic_executor = SymbolicExecutor()

        self.calculations = EvaluatorsCalculations()

    def full_analysis(self) -> Dict[str, Any]:  
        """Orchestrate multi-layered analysis"""  
        issues = []
        for filepath in self._discover_code_files():
            tree = self.ast_analyzer.parse_file(filepath)
            if tree is None:  # Handle parse failures
                logger.warning(f"Skipping analysis for {filepath} due to parse error")
                continue
                
            issues.extend(self.ast_analyzer.detect_anti_patterns(tree))
            
            # Symbolic execution for security-critical functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and "security" in node.name.lower():
                    issues.extend(
                        self.symbolic_executor.analyze_security_constraints(node)
                    )
        return {
            "technical_debt": self.calculations.calculate_debt(issues),
            "remediation_plan": self.calculations.prioritize_remediation(issues),
            "security_metrics": self._aggregate_security_stats(issues)
        }
    
    def _discover_code_files(self) -> List[str]:
        python_files = []
        for root, _, files in os.walk(self.ast_analyzer.codebase_path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def _aggregate_security_stats(self, issues: List[Dict]) -> Dict[str, Any]:
        """Aggregate security issues by type, severity, and count."""
        summary = {
            "total_issues": 0,
            "critical_count": 0,
            "issue_types": {},
            "max_severity": 0.0
        }
    
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            severity = issue.get("severity", 0.0)
    
            summary["total_issues"] += 1
            summary["max_severity"] = max(summary["max_severity"], severity)
    
            if severity >= 0.8:
                summary["critical_count"] += 1
    
            if issue_type not in summary["issue_types"]:
                summary["issue_types"][issue_type] = 0
            summary["issue_types"][issue_type] += 1
    
        return summary


class ASTAnalyzer:  
    def __init__(self, codebase_path: str):  
        self.codebase_path = codebase_path  
        self.ast_cache: Dict[str, ast.AST] = {}  # Filepath → Parsed AST  

    def parse_file(self, filepath: str) -> Optional[ast.AST]:  
        """Build AST with type inference and scope tracking"""  
        try:  
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read())
                self.ast_cache[filepath] = tree  
                return tree  
        except Exception as e:  
            logger.error(f"Failed to parse {filepath}: {e}")  
            return None  

    def detect_anti_patterns(self, tree: ast.AST) -> List[Dict]:  
        """Identify complex anti-patterns using AST traversal"""  
        patterns = []  
        for node in ast.walk(tree):  
            # Example: Detect nested loops exceeding cognitive complexity threshold  
            if isinstance(node, ast.For) and any(isinstance(parent, ast.For) for parent in ast.iter_child_nodes(node)):  
                patterns.append({
                    "type": "nested_loop",
                    "line": node.lineno,
                    "severity": 0.9,
                    "cognitive_complexity": self._calculate_cognitive_complexity(node)
                })  
        return patterns
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> float:
        """Estimate cognitive complexity by analyzing nesting and control structures"""
        complexity = 0
        stack = [(node, 0)]
    
        while stack:
            current, depth = stack.pop()
            if isinstance(current, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                complexity += 1 + depth  # increase for each nested control structure
            elif isinstance(current, ast.Call):
                complexity += 0.5  # function calls add moderate complexity
    
            for child in ast.iter_child_nodes(current):
                stack.append((child, depth + 1))
    
        return round(complexity, 2)

class SymbolicExecutor:  
    def __init__(self):  
        self.constraint_log = []
        self.taint_sources = {"input", "request", "os.getenv", "sys.argv"}
        self.dangerous_sinks = {"eval", "exec", "os.system", "subprocess.call", "open"}

    def analyze_security_constraints(self, func_ast: ast.FunctionDef) -> List[Dict]:
        """
        Perform static checks for security-sensitive function patterns:
        - Use of eval()
        - Unparameterized SQL execution
        - Dangerous exec() usage
        """
        issues = []

        for node in ast.walk(func_ast):
            # Check for eval()
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "eval":
                issues.append({
                    "type": "unsafe_eval",
                    "line": node.lineno,
                    "severity": 0.9,
                    "description": "Use of eval() detected, which can lead to code injection"
                })

            # Check for exec()
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "exec":
                issues.append({
                    "type": "unsafe_exec",
                    "line": node.lineno,
                    "severity": 0.8,
                    "description": "Use of exec() detected, which is a potential security risk"
                })

            # Check for raw SQL query string in execute()
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "execute":
                    if node.args and isinstance(node.args[0], ast.BinOp):
                        issues.append({
                            "type": "unparameterized_sql",
                            "line": node.lineno,
                            "severity": 1.0,
                            "description": "Unparameterized SQL query detected; potential SQL injection"
                        })

        return issues

    def _extract_constraints(self, node: ast.AST) -> List[Dict[str, Any]]:
        """Extract symbolic constraints from conditions like 'if', 'while', and function calls."""
        constraints = []
    
        for sub in ast.walk(node):
            if isinstance(sub, ast.If):
                condition = ast.dump(sub.test)
                constraints.append({
                    "type": "branch_condition",
                    "line": sub.lineno,
                    "condition": condition
                })
    
            elif isinstance(sub, ast.While):
                condition = ast.dump(sub.test)
                constraints.append({
                    "type": "loop_condition",
                    "line": sub.lineno,
                    "condition": condition
                })
    
            elif isinstance(sub, ast.Compare):
                if hasattr(sub, "left") and hasattr(sub, "comparators"):
                    left = ast.unparse(sub.left) if hasattr(ast, "unparse") else ast.dump(sub.left)
                    right = ast.unparse(sub.comparators[0]) if hasattr(ast, "unparse") else ast.dump(sub.comparators[0])
                    op = type(sub.ops[0]).__name__
                    constraints.append({
                        "type": "comparison",
                        "line": sub.lineno,
                        "expression": f"{left} {op} {right}"
                    })
    
            elif isinstance(sub, ast.Call):
                func_name = self._resolve_func_name(sub.func)
                if func_name in self.dangerous_sinks:
                    constraints.append({
                        "type": "sink_call",
                        "line": sub.lineno,
                        "sink": func_name,
                        "tainted_args": [
                            ast.unparse(arg) if hasattr(ast, "unparse") else ast.dump(arg)
                            for arg in sub.args if self._is_potentially_tainted(arg)
                        ]
                    })
    
        return constraints

    def _resolve_func_name(self, func) -> str:
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return f"{self._resolve_func_name(func.value)}.{func.attr}"
        return "unknown"

    def _is_potentially_tainted(self, node: ast.AST) -> bool:
        """Crude check to see if a node may involve a taint source."""
        if isinstance(node, ast.Name):
            return node.id in self.taint_sources
        elif isinstance(node, ast.Call):
            func_name = self._resolve_func_name(node.func)
            return func_name in self.taint_sources
        return False
    

#class TechnicalDebtCalculator:

class DataFlowAnalyzer:
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.call_graph = nx.DiGraph()
        self.data_dependencies = nx.MultiDiGraph()
        self.def_use_chains = {}

    def build_call_graph(self):
        """Build basic call graph using AST (no inter-procedural analysis)"""
        for filepath in self._discover_code_files():
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    caller = node.name
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and hasattr(child.func, "id"):
                            callee = child.func.id
                            self.call_graph.add_edge(caller, callee)

    def _discover_code_files(self) -> List[str]:
        python_files = []
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    def track_data_flow(self):
        """Extended data flow analysis across function boundaries"""
        for filepath in self._discover_code_files():
            tree = self.ast_analyzer.parse_file(filepath)
            visitor = DataFlowVisitor(self.call_graph)
            visitor.visit(tree)
            self.data_dependencies = nx.compose(
                self.data_dependencies,
                visitor.dependencies
            )
            self.def_use_chains.update(visitor.def_use_chains)

    def detect_interprocedural_issues(self) -> List[Dict]:
        """Find cross-function data flow vulnerabilities"""
        issues = []
        for node in self.data_dependencies.nodes:
            if self._is_unvalidated_input(node):
                issues.append({
                    "type": "unvalidated_data_flow",
                    "severity": "critical",
                    "source": node.source,
                    "sink": node.sink,
                    "data_types": node.types
                })
        return issues
    
    def _is_unvalidated_input(self, node):
        return []

class DataFlowVisitor(ast.NodeVisitor):
    def __init__(self, call_graph):
        self.call_graph = call_graph
        self.current_function = None
        self.dependencies = nx.MultiDiGraph()
        self.def_use_chains = {}
        
    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.generic_visit(node)
        
    def visit_Call(self, node):
        callee = self._resolve_callee(node.func)
        if callee in self.call_graph:
            self._analyze_arguments(node, callee)
            
    def _analyze_arguments(self, node, callee):
        # Map argument data flow between caller and callee
        for idx, arg in enumerate(node.args):
            var_name = self._extract_variable(arg)
            if var_name:
                self.dependencies.add_edge(
                    self.current_function,
                    callee,
                    arg_index=idx,
                    var_name=var_name
                )

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Behavioral Validator ===\n")
    import json
    codebase_path = "src/agents/evaluators/"

    analyzer = StaticAnalyzer(codebase_path=codebase_path)
    logger.info(f"{analyzer}")
    logger.info("Analyzed Report:\n" + json.dumps(analyzer.full_analysis(), indent=4))
    print(json.dumps (analyzer.full_analysis(), indent=4))

    print(f"\n* * * * * Phase 2 * * * * *\n")
    data = DataFlowAnalyzer(codebase_path=codebase_path)
    logger.info(f"{data}")
    logger.info("Analyzed Report:\n" + json.dumps(data.build_call_graph(), indent=4))
    print(json.dumps (data.build_call_graph(), indent=4))

    print(f"\n* * * * * Phase 3 * * * * *\n")
    print("\n=== Successfully Ran Behavioral Validator ===\n")
