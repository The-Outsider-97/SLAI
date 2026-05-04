from __future__ import annotations

import ast
import hashlib
import os
import networkx as nx

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .config_loader import load_global_config, get_config_section
from .evaluation_errors import ConfigLoadError, OperationalError, ValidationFailureError
from .evaluators_calculations import EvaluatorsCalculations
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Static Analyzer")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AnalysisIssue:
    """Structured static-analysis issue suitable for remediation workflows."""

    type: str
    severity: float
    file_path: str
    line: int
    category: str
    description: str
    rule_id: str
    estimated_fix_time: float = 1.0
    column: Optional[int] = None
    symbol: Optional[str] = None
    remediation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValidationFailureError("analysis_issue.type", self.type, "non-empty string")
        self.type = self.type.strip()

        if not isinstance(self.category, str) or not self.category.strip():
            raise ValidationFailureError("analysis_issue.category", self.category, "non-empty string")
        self.category = self.category.strip()

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValidationFailureError(
                "analysis_issue.description",
                self.description,
                "non-empty string",
            )
        self.description = self.description.strip()

        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValidationFailureError("analysis_issue.rule_id", self.rule_id, "non-empty string")
        self.rule_id = self.rule_id.strip()

        if not isinstance(self.file_path, str) or not self.file_path.strip():
            raise ValidationFailureError("analysis_issue.file_path", self.file_path, "non-empty string")
        self.file_path = self.file_path.strip()

        if not isinstance(self.line, int) or self.line < 1:
            raise ValidationFailureError("analysis_issue.line", self.line, "positive integer")

        if self.column is not None and (not isinstance(self.column, int) or self.column < 0):
            raise ValidationFailureError("analysis_issue.column", self.column, "non-negative integer")

        self.severity = _clamp_severity(self.severity)
        self.estimated_fix_time = max(float(self.estimated_fix_time), 0.1)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParseResult:
    """AST parse result with error capture for resilient orchestration."""

    file_path: str
    tree: Optional[ast.AST]
    error: Optional[str] = None


@dataclass(slots=True)
class FileAnalysisResult:
    """Per-file static-analysis result."""

    file_path: str
    parsed: bool
    issue_count: int
    issues: List[Dict[str, Any]]
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    max_cyclomatic_complexity: float = 0.0
    max_cognitive_complexity: float = 0.0
    duplicate_fingerprints: List[str] = field(default_factory=list)
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StaticAnalysisReport:
    """Top-level analysis report returned by the static analyzer."""

    codebase_path: str
    analyzed_at: str
    configuration: Dict[str, Any]
    summary: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    security_metrics: Dict[str, Any]
    technical_debt: float
    remediation_plan: List[Dict[str, Any]]
    issues: List[Dict[str, Any]]
    files: List[Dict[str, Any]]
    call_graph: Dict[str, Any]
    data_flow: Dict[str, Any]
    parse_failures: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class StaticAnalyzer:
    """
    Production-oriented static analysis orchestrator.

    Responsibilities
    ----------------
    - Parse and analyze Python source files across a codebase
    - Detect quality and security-oriented AST anti-patterns
    - Build call-graph and lightweight data-flow information
    - Integrate with EvaluatorsCalculations for debt and remediation ranking
    - Produce a structured report suitable for automation and reporting layers
    """

    def __init__(self, codebase_path: str) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))

        if not isinstance(codebase_path, str) or not codebase_path.strip():
            raise OperationalError(
                "StaticAnalyzer requires a non-empty codebase_path.",
                context={"codebase_path": codebase_path},
            )

        self.codebase_path = Path(codebase_path).expanduser().resolve()
        if not self.codebase_path.exists() or not self.codebase_path.is_dir():
            raise OperationalError(
                "Configured codebase path does not exist or is not a directory.",
                context={"codebase_path": str(self.codebase_path)},
            )

        self.validation_protocol = get_config_section("validation_protocol")
        static_analysis_config = self.validation_protocol.get("static_analysis", {})
        if not isinstance(static_analysis_config, Mapping):
            raise ConfigLoadError(self.config_path, "validation_protocol.static_analysis", "section must be a mapping")

        self.static_config = dict(static_analysis_config)
        self.security_config = dict(self.static_config.get("security", {}))
        code_quality_config = self.static_config.get("code_quality", {})
        self.code_quality_config = dict(code_quality_config if isinstance(code_quality_config, Mapping) else {})
        self.complexity_config = dict(self.code_quality_config.get("complexity", {}))
        self.data_flow_config = dict(get_config_section("data_flow_analysis") or {})

        self.enabled = bool(self.static_config.get("enable", True))
        self.max_cyclomatic = self._require_positive_number(
            self.complexity_config.get("cyclomatic", 15),
            "validation_protocol.static_analysis.code_quality.complexity.cyclomatic",
        )
        self.max_cognitive = self._require_positive_number(
            self.complexity_config.get("cognitive", 20),
            "validation_protocol.static_analysis.code_quality.complexity.cognitive",
        )
        self.max_call_depth = self._require_positive_integer(
            self.data_flow_config.get("max_call_depth", 10),
            "data_flow_analysis.max_call_depth",
        )
        self.track_types = bool(self.data_flow_config.get("track_types", True))
        self.analyze_taint = bool(self.data_flow_config.get("analyze_taint", True))
        self.vulnerability_patterns = self._normalize_string_list(
            self.data_flow_config.get(
                "vulnerability_patterns",
                ["sql_injection", "xss", "path_traversal"],
            ),
            "data_flow_analysis.vulnerability_patterns",
        )
        self.max_critical = int(self.security_config.get("max_critical", 0))
        self.max_high = int(self.security_config.get("max_high", 3))

        self.ast_analyzer = ASTAnalyzer(
            codebase_path=str(self.codebase_path),
            max_cyclomatic=self.max_cyclomatic,
            max_cognitive=self.max_cognitive,
        )
        self.symbolic_executor = SymbolicExecutor(analyze_taint=self.analyze_taint)
        self.data_flow_analyzer = DataFlowAnalyzer(
            codebase_path=str(self.codebase_path),
            ast_analyzer=self.ast_analyzer,
            max_call_depth=self.max_call_depth,
            analyze_taint=self.analyze_taint,
        )
        self.calculations = EvaluatorsCalculations()

        logger.info("Static Analyzer successfully initialized for %s", self.codebase_path)

    def full_analysis(self) -> Dict[str, Any]:
        """Run a complete multi-layer analysis and return a structured report."""
        if not self.enabled:
            logger.warning("Static analysis is disabled by configuration")
            return StaticAnalysisReport(
                codebase_path=str(self.codebase_path),
                analyzed_at=_utcnow().isoformat(),
                configuration=self._configuration_snapshot(),
                summary={
                    "enabled": False,
                    "files_discovered": 0,
                    "files_analyzed": 0,
                    "files_skipped": 0,
                    "total_issues": 0,
                },
                quality_metrics={},
                security_metrics={},
                technical_debt=0.0,
                remediation_plan=[],
                issues=[],
                files=[],
                call_graph={},
                data_flow={},
                parse_failures=[],
            ).to_dict()

        issues: List[AnalysisIssue] = []
        file_results: List[FileAnalysisResult] = []
        parse_failures: List[Dict[str, str]] = []
        function_fingerprint_index: Dict[str, List[Tuple[str, ast.AST]]] = {}

        filepaths = self._discover_code_files()
        for filepath in filepaths:
            file_result, trees, fingerprints = self._analyze_single_file(filepath)
            file_results.append(file_result)
            issues.extend(trees["issues"])

            if file_result.parse_error:
                parse_failures.append({"file_path": file_result.file_path, "error": file_result.parse_error})

            for fingerprint, function_node in fingerprints:
                function_fingerprint_index.setdefault(fingerprint, []).append((file_result.file_path, function_node))

        issues.extend(self._detect_duplicate_functions(function_fingerprint_index))

        call_graph = self.data_flow_analyzer.build_call_graph()
        self.data_flow_analyzer.track_data_flow()
        issues.extend(self.data_flow_analyzer.detect_interprocedural_issues())

        issue_payloads = [issue.to_dict() for issue in sorted(issues, key=self._issue_sort_key)]
        security_metrics = self._aggregate_security_stats(issue_payloads)
        quality_metrics = self._aggregate_quality_metrics(issue_payloads, file_results, parse_failures)
        technical_debt = self.calculations.calculate_debt(issue_payloads) if issue_payloads else 0.0
        remediation_plan = self.calculations.prioritize_remediation(issue_payloads) if issue_payloads else []

        report = StaticAnalysisReport(
            codebase_path=str(self.codebase_path),
            analyzed_at=_utcnow().isoformat(),
            configuration=self._configuration_snapshot(),
            summary={
                "enabled": True,
                "files_discovered": len(filepaths),
                "files_analyzed": sum(1 for result in file_results if result.parsed),
                "files_skipped": sum(1 for result in file_results if not result.parsed),
                "total_issues": len(issue_payloads),
                "critical_issues": security_metrics.get("critical_count", 0),
                "high_severity_issues": security_metrics.get("high_severity_count", 0),
                "security_threshold_status": self._security_threshold_status(security_metrics),
            },
            quality_metrics=quality_metrics,
            security_metrics=security_metrics,
            technical_debt=technical_debt,
            remediation_plan=remediation_plan,
            issues=issue_payloads,
            files=[result.to_dict() for result in file_results],
            call_graph=self.data_flow_analyzer.get_call_graph_summary(call_graph),
            data_flow=self.data_flow_analyzer.get_data_flow_summary(),
            parse_failures=parse_failures,
        )
        return report.to_dict()

    def _analyze_single_file(
        self,
        filepath: str,
    ) -> Tuple[FileAnalysisResult, Dict[str, Any], List[Tuple[str, ast.AST]]]:
        parse_result = self.ast_analyzer.parse_file(filepath)
        if parse_result.tree is None:
            result = FileAnalysisResult(
                file_path=filepath,
                parsed=False,
                issue_count=0,
                issues=[],
                parse_error=parse_result.error,
            )
            return result, {"issues": []}, []

        anti_pattern_issues, file_metrics, fingerprints = self.ast_analyzer.detect_anti_patterns(parse_result.tree, filepath)

        security_issues: List[AnalysisIssue] = []
        for node in ast.walk(parse_result.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                security_issues.extend(self.symbolic_executor.analyze_security_constraints(node, filepath))

        issues = anti_pattern_issues + security_issues
        result = FileAnalysisResult(
            file_path=filepath,
            parsed=True,
            issue_count=len(issues),
            issues=[issue.to_dict() for issue in issues],
            function_count=file_metrics["function_count"],
            class_count=file_metrics["class_count"],
            import_count=file_metrics["import_count"],
            max_cyclomatic_complexity=file_metrics["max_cyclomatic_complexity"],
            max_cognitive_complexity=file_metrics["max_cognitive_complexity"],
            duplicate_fingerprints=[item[0] for item in fingerprints],
        )
        return result, {"issues": issues}, fingerprints

    def _discover_code_files(self) -> List[str]:
        python_files: List[str] = []
        for root, _, files in os.walk(self.codebase_path):
            for file_name in files:
                if file_name.endswith(".py"):
                    python_files.append(str(Path(root) / file_name))
        return sorted(python_files)

    def _detect_duplicate_functions(
        self,
        function_fingerprint_index: Mapping[str, Sequence[Tuple[str, ast.AST]]],
    ) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []
        for fingerprint, matches in function_fingerprint_index.items():
            if len(matches) < 2:
                continue

            match_paths = sorted({path for path, _ in matches})
            for filepath, function_node in matches:
                issues.append(
                    AnalysisIssue(
                        type="duplicate_code",
                        severity=0.7,
                        file_path=filepath,
                        line=getattr(function_node, "lineno", 1),
                        column=getattr(function_node, "col_offset", 0),
                        category="quality",
                        description="Function body duplicates logic found in another location.",
                        rule_id="SA-DUP-001",
                        symbol=getattr(function_node, "name", None),
                        remediation="Extract shared logic into a reusable helper or consolidate duplicate implementations.",
                        estimated_fix_time=2.5,
                        metadata={
                            "fingerprint": fingerprint,
                            "duplicate_locations": match_paths,
                        },
                    )
                )
        return issues

    def _aggregate_security_stats(self, issues: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        summary = {
            "total_issues": 0,
            "critical_count": 0,
            "high_severity_count": 0,
            "issue_types": {},
            "category_counts": {},
            "max_severity": 0.0,
        }

        for issue in issues:
            issue_type = str(issue.get("type", "unknown"))
            category = str(issue.get("category", "uncategorized"))
            severity = _clamp_severity(issue.get("severity", 0.0))

            summary["total_issues"] += 1
            summary["max_severity"] = max(summary["max_severity"], severity)
            summary["issue_types"][issue_type] = summary["issue_types"].get(issue_type, 0) + 1
            summary["category_counts"][category] = summary["category_counts"].get(category, 0) + 1

            if severity >= 0.9:
                summary["critical_count"] += 1
            if severity >= 0.75:
                summary["high_severity_count"] += 1

        summary["critical_limit_exceeded"] = summary["critical_count"] > self.max_critical
        summary["high_limit_exceeded"] = summary["high_severity_count"] > self.max_high
        return summary

    def _aggregate_quality_metrics(
        self,
        issues: Sequence[Mapping[str, Any]],
        file_results: Sequence[FileAnalysisResult],
        parse_failures: Sequence[Mapping[str, str]],
    ) -> Dict[str, Any]:
        function_count = sum(result.function_count for result in file_results)
        class_count = sum(result.class_count for result in file_results)
        import_count = sum(result.import_count for result in file_results)
        parsed_files = [result for result in file_results if result.parsed]

        max_cyclomatic = max((result.max_cyclomatic_complexity for result in parsed_files), default=0.0)
        max_cognitive = max((result.max_cognitive_complexity for result in parsed_files), default=0.0)
        issue_density = (len(issues) / len(parsed_files)) if parsed_files else 0.0

        by_rule: Dict[str, int] = {}
        for issue in issues:
            rule_id = str(issue.get("rule_id", "unknown"))
            by_rule[rule_id] = by_rule.get(rule_id, 0) + 1

        return {
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
            "issue_density_per_file": issue_density,
            "max_cyclomatic_complexity": max_cyclomatic,
            "max_cognitive_complexity": max_cognitive,
            "parse_failure_count": len(parse_failures),
            "issues_by_rule": dict(sorted(by_rule.items())),
        }

    def _security_threshold_status(self, security_metrics: Mapping[str, Any]) -> str:
        critical_exceeded = bool(security_metrics.get("critical_limit_exceeded", False))
        high_exceeded = bool(security_metrics.get("high_limit_exceeded", False))
        if critical_exceeded or high_exceeded:
            return "Threshold Exceeded"
        return "Within Thresholds"

    def _configuration_snapshot(self) -> Dict[str, Any]:
        return {
            "config_path": self.config_path,
            "static_analysis_enabled": self.enabled,
            "max_cyclomatic": self.max_cyclomatic,
            "max_cognitive": self.max_cognitive,
            "max_call_depth": self.max_call_depth,
            "analyze_taint": self.analyze_taint,
            "track_types": self.track_types,
            "vulnerability_patterns": list(self.vulnerability_patterns),
            "max_critical": self.max_critical,
            "max_high": self.max_high,
        }

    @staticmethod
    def _issue_sort_key(issue: AnalysisIssue) -> Tuple[float, str, int, str]:
        return (-issue.severity, issue.file_path, issue.line, issue.type)

    @staticmethod
    def _require_positive_number(value: Any, field_name: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError("<config>", field_name, f"expected numeric value, got {value!r}") from exc
        if numeric <= 0:
            raise ConfigLoadError("<config>", field_name, "must be greater than zero")
        return numeric

    @staticmethod
    def _require_positive_integer(value: Any, field_name: str) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError("<config>", field_name, f"expected integer value, got {value!r}") from exc
        if numeric <= 0:
            raise ConfigLoadError("<config>", field_name, "must be greater than zero")
        return numeric

    @staticmethod
    def _normalize_string_list(value: Any, field_name: str) -> List[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ConfigLoadError("<config>", field_name, "must be a sequence of strings")
        normalized: List[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ConfigLoadError("<config>", field_name, "contains invalid string value")
            normalized.append(item.strip())
        return normalized


# ---------------------------------------------------------------------------
# AST quality analysis
# ---------------------------------------------------------------------------
class ASTAnalyzer:
    """AST-based anti-pattern and maintainability analyzer."""

    CONTROL_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Match,
    )

    BRANCH_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.IfExp,
        ast.Match,
    )

    def __init__(self, codebase_path: str, max_cyclomatic: float, max_cognitive: float) -> None:
        self.codebase_path = str(Path(codebase_path).resolve())
        self.max_cyclomatic = float(max_cyclomatic)
        self.max_cognitive = float(max_cognitive)
        self.ast_cache: Dict[str, ast.AST] = {}
        self.parse_errors: Dict[str, str] = {}

    def parse_file(self, filepath: str) -> ParseResult:
        """Parse a Python file into an AST and annotate parent pointers."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
                source = handle.read()
            tree = ast.parse(source, filename=filepath)
            self._annotate_parents(tree)
            self.ast_cache[filepath] = tree
            self.parse_errors.pop(filepath, None)
            return ParseResult(file_path=filepath, tree=tree)
        except SyntaxError as exc:
            message = f"SyntaxError: {exc.msg} (line {exc.lineno}, column {exc.offset})"
            self.parse_errors[filepath] = message
            logger.error("Failed to parse %s: %s", filepath, message)
            return ParseResult(file_path=filepath, tree=None, error=message)
        except OSError as exc:
            message = f"OSError: {exc}"
            self.parse_errors[filepath] = message
            logger.error("Failed to read %s: %s", filepath, message)
            return ParseResult(file_path=filepath, tree=None, error=message)

    def detect_anti_patterns(self, tree: ast.AST, filepath: str,
    ) -> Tuple[List[AnalysisIssue], Dict[str, Any], List[Tuple[str, ast.AST]]]:
        issues: List[AnalysisIssue] = []
        function_count = 0
        class_count = 0
        import_count = 0
        max_cyclomatic = 0.0
        max_cognitive = 0.0
        fingerprints: List[Tuple[str, ast.AST]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
                if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                    issues.append(
                        self._make_issue(
                            filepath,
                            node,
                            issue_type="duplicate_code",
                            category="quality",
                            rule_id="SA-IMP-001",
                            description="Wildcard import reduces clarity and hinders static reasoning.",
                            remediation="Replace wildcard imports with explicit imports.",
                            severity=0.45,
                            estimated_fix_time=0.5,
                        )
                    )

            if isinstance(node, ast.ClassDef):
                class_count += 1

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                cyclomatic = self._calculate_cyclomatic_complexity(node)
                cognitive = self._calculate_cognitive_complexity(node)
                max_cyclomatic = max(max_cyclomatic, cyclomatic)
                max_cognitive = max(max_cognitive, cognitive)
                fingerprints.append((self._fingerprint_function(node), node))

                if cyclomatic > self.max_cyclomatic:
                    issues.append(
                        self._make_issue(
                            filepath,
                            node,
                            issue_type="nested_control",
                            category="quality",
                            rule_id="SA-CYC-001",
                            description=f"Function exceeds cyclomatic complexity threshold ({cyclomatic:.1f} > {self.max_cyclomatic:.1f}).",
                            remediation="Split the function into smaller units and reduce branching paths.",
                            severity=_scaled_severity(cyclomatic, self.max_cyclomatic),
                            estimated_fix_time=3.0,
                            symbol=node.name,
                            metadata={"cyclomatic_complexity": cyclomatic},
                        )
                    )

                if cognitive > self.max_cognitive:
                    issues.append(
                        self._make_issue(
                            filepath,
                            node,
                            issue_type="nested_control",
                            category="quality",
                            rule_id="SA-COG-001",
                            description=f"Function exceeds cognitive complexity threshold ({cognitive:.1f} > {self.max_cognitive:.1f}).",
                            remediation="Reduce nesting depth and extract condition-heavy logic into helpers.",
                            severity=_scaled_severity(cognitive, self.max_cognitive),
                            estimated_fix_time=3.5,
                            symbol=node.name,
                            metadata={"cognitive_complexity": cognitive},
                        )
                    )

                if self._contains_nested_loop(node):
                    issues.append(
                        self._make_issue(
                            filepath,
                            node,
                            issue_type="nested_loop",
                            category="quality",
                            rule_id="SA-LOOP-001",
                            description="Nested iterative control flow detected.",
                            remediation="Refactor nested iteration or precompute lookup structures to reduce complexity.",
                            severity=0.8,
                            estimated_fix_time=2.5,
                            symbol=node.name,
                        )
                    )

                if self._has_law_of_demeter_violation(node):
                    issues.append(
                        self._make_issue(
                            filepath,
                            node,
                            issue_type="violation_of_law_of_demeter",
                            category="quality",
                            rule_id="SA-LOD-001",
                            description="Deep attribute chaining indicates tight coupling and brittle navigation.",
                            remediation="Introduce intermediate abstractions or helper methods to reduce traversal depth.",
                            severity=0.6,
                            estimated_fix_time=1.5,
                            symbol=node.name,
                        )
                    )

                if len(node.args.args) > 6:
                    issues.append(
                        self._make_issue(
                            filepath,
                            node,
                            issue_type="nested_control",
                            category="quality",
                            rule_id="SA-ARG-001",
                            description=f"Function has an excessive number of positional parameters ({len(node.args.args)}).",
                            remediation="Group related parameters into value objects or configuration mappings.",
                            severity=0.5,
                            estimated_fix_time=1.5,
                            symbol=node.name,
                            metadata={"argument_count": len(node.args.args)},
                        )
                    )

            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None or (isinstance(handler.type, ast.Name) and handler.type.id == "Exception"):
                        issues.append(
                            self._make_issue(
                                filepath,
                                handler,
                                issue_type="security_risk",
                                category="security",
                                rule_id="SA-EXC-001",
                                description="Broad exception handling masks root causes and weakens failure isolation.",
                                remediation="Catch specific exception classes and handle them explicitly.",
                                severity=0.55,
                                estimated_fix_time=1.0,
                            )
                        )

        metrics = {
            "function_count": function_count,
            "class_count": class_count,
            "import_count": import_count,
            "max_cyclomatic_complexity": round(max_cyclomatic, 2),
            "max_cognitive_complexity": round(max_cognitive, 2),
        }
        return issues, metrics, fingerprints

    def _annotate_parents(self, tree: ast.AST) -> None:
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, "_parent", parent)

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> float:
        complexity = 1.0
        for subnode in ast.walk(node):
            if isinstance(subnode, self.BRANCH_NODES):
                complexity += 1.0
            elif isinstance(subnode, ast.BoolOp):
                complexity += max(0, len(subnode.values) - 1)
            elif isinstance(subnode, ast.comprehension):
                complexity += 1.0
        return float(complexity)

    def _calculate_cognitive_complexity(self, node: ast.AST) -> float:
        def walk(current: ast.AST, nesting: int) -> float:
            score = 0.0
            for child in ast.iter_child_nodes(current):
                additional_nesting = nesting
                if isinstance(child, self.CONTROL_NODES):
                    score += 1.0 + nesting
                    additional_nesting = nesting + 1
                elif isinstance(child, ast.BoolOp):
                    score += max(0, len(child.values) - 1) * 0.5
                elif isinstance(child, ast.Call):
                    score += 0.1
                score += walk(child, additional_nesting)
            return score

        return round(walk(node, 0), 2)

    def _contains_nested_loop(self, node: ast.AST) -> bool:
        loop_nodes = (ast.For, ast.AsyncFor, ast.While)
        for subnode in ast.walk(node):
            if isinstance(subnode, loop_nodes):
                for child in ast.iter_child_nodes(subnode):
                    if isinstance(child, loop_nodes):
                        return True
                    if any(isinstance(descendant, loop_nodes) for descendant in ast.walk(child) if descendant is not subnode):
                        return True
        return False

    def _has_law_of_demeter_violation(self, node: ast.AST) -> bool:
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Attribute):
                if self._attribute_depth(subnode) >= 4:
                    return True
        return False

    def _attribute_depth(self, node: ast.Attribute) -> int:
        depth = 1
        current: Any = node.value
        while isinstance(current, ast.Attribute):
            depth += 1
            current = current.value
        return depth

    def _fingerprint_function(self, node: ast.AST) -> str:
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _make_issue(
        self,
        filepath: str,
        node: ast.AST,
        issue_type: str,
        category: str,
        rule_id: str,
        description: str,
        remediation: str,
        severity: float,
        estimated_fix_time: float,
        symbol: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AnalysisIssue:
        return AnalysisIssue(
            type=issue_type,
            severity=severity,
            file_path=filepath,
            line=int(getattr(node, "lineno", 1)),
            column=int(getattr(node, "col_offset", 0)),
            category=category,
            description=description,
            rule_id=rule_id,
            estimated_fix_time=estimated_fix_time,
            symbol=symbol,
            remediation=remediation,
            metadata=dict(metadata or {}),
        )


# ---------------------------------------------------------------------------
# Security and symbolic inspection
# ---------------------------------------------------------------------------
class SymbolicExecutor:
    """
    Lightweight symbolic and taint-oriented executor.

    This class intentionally remains static and conservative. It does not claim
    full symbolic execution, but it extracts branch constraints and identifies
    potentially dangerous sink usage with tainted input propagation heuristics.
    """

    def __init__(self, analyze_taint: bool = True) -> None:
        self.analyze_taint = analyze_taint
        self.constraint_log: List[Dict[str, Any]] = []
        self.taint_sources = {
            "input",
            "sys.argv",
            "os.getenv",
            "request.args.get",
            "request.form.get",
            "request.get_json",
            "request.json.get",
        }
        self.validation_functions = {"sanitize", "validate", "escape", "quote", "clean"}
        self.dangerous_sinks = {
            "eval",
            "exec",
            "os.system",
            "subprocess.call",
            "subprocess.run",
            "subprocess.Popen",
            "pickle.loads",
            "yaml.load",
            "open",
        }

    def analyze_security_constraints(
        self,
        func_ast: ast.AST,
        filepath: str,
    ) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []
        constraints = self._extract_constraints(func_ast)
        self.constraint_log.extend(constraints)

        tainted_variables = self._infer_tainted_variables(func_ast) if self.analyze_taint else set()
        function_name = getattr(func_ast, "name", None)

        for node in ast.walk(func_ast):
            if not isinstance(node, ast.Call):
                continue

            function_name_resolved = self._resolve_func_name(node.func)
            if function_name_resolved == "eval":
                issues.append(
                    self._issue(
                        filepath,
                        node,
                        issue_type="security_risk",
                        rule_id="SA-SEC-001",
                        description="Use of eval() detected; this can enable arbitrary code execution.",
                        remediation="Replace eval() with safer parsing or explicit dispatch logic.",
                        severity=0.95,
                        symbol=function_name,
                    )
                )
            elif function_name_resolved == "exec":
                issues.append(
                    self._issue(
                        filepath,
                        node,
                        issue_type="security_risk",
                        rule_id="SA-SEC-002",
                        description="Use of exec() detected; dynamic execution is a high-risk pattern.",
                        remediation="Avoid exec() and use explicit APIs or controlled command handlers.",
                        severity=0.9,
                        estimated_fix_time=2.0,
                        symbol=function_name,
                    )
                )
            elif function_name_resolved in {"os.system", "subprocess.call", "subprocess.run", "subprocess.Popen"}:
                if self._call_uses_tainted_arguments(node, tainted_variables):
                    issues.append(
                        self._issue(
                            filepath,
                            node,
                            issue_type="security_risk",
                            rule_id="SA-SEC-003",
                            description="Tainted data flows into shell or subprocess execution.",
                            remediation="Validate or sanitize command arguments and avoid shell-based execution where possible.",
                            severity=1.0,
                            estimated_fix_time=2.5,
                            symbol=function_name,
                            metadata={"sink": function_name_resolved},
                        )
                    )
            elif function_name_resolved.endswith(".execute"):
                if self._is_dynamic_sql(node) or self._call_uses_tainted_arguments(node, tainted_variables):
                    issues.append(
                        self._issue(
                            filepath,
                            node,
                            issue_type="security_risk",
                            rule_id="SA-SEC-004",
                            description="Potentially unparameterized SQL execution detected.",
                            remediation="Use parameterized queries and keep user input separate from SQL text.",
                            severity=1.0,
                            estimated_fix_time=2.0,
                            symbol=function_name,
                            metadata={"sink": function_name_resolved},
                        )
                    )
            elif function_name_resolved == "yaml.load":
                if not self._uses_safe_loader(node):
                    issues.append(
                        self._issue(
                            filepath,
                            node,
                            issue_type="security_risk",
                            rule_id="SA-SEC-005",
                            description="yaml.load() detected without an explicit safe loader.",
                            remediation="Use yaml.safe_load() or pass a safe Loader explicitly.",
                            severity=0.8,
                            estimated_fix_time=1.0,
                            symbol=function_name,
                        )
                    )
            elif function_name_resolved == "pickle.loads":
                issues.append(
                    self._issue(
                        filepath,
                        node,
                        issue_type="security_risk",
                        rule_id="SA-SEC-006",
                        description="pickle.loads() may deserialize untrusted data unsafely.",
                        remediation="Avoid pickle for untrusted payloads and use safer serialization formats.",
                        severity=0.85,
                        estimated_fix_time=1.5,
                        symbol=function_name,
                    )
                )
            elif function_name_resolved == "open":
                if self._call_uses_tainted_arguments(node, tainted_variables):
                    issues.append(
                        self._issue(
                            filepath,
                            node,
                            issue_type="security_risk",
                            rule_id="SA-SEC-007",
                            description="Potential tainted file path reaches open().",
                            remediation="Validate and normalize filesystem paths before opening files.",
                            severity=0.75,
                            estimated_fix_time=1.5,
                            symbol=function_name,
                        )
                    )

        return issues

    def _extract_constraints(self, node: ast.AST) -> List[Dict[str, Any]]:
        constraints: List[Dict[str, Any]] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.If):
                constraints.append(
                    {
                        "type": "branch_condition",
                        "line": getattr(sub, "lineno", 1),
                        "condition": ast.dump(sub.test, include_attributes=False),
                    }
                )
            elif isinstance(sub, ast.While):
                constraints.append(
                    {
                        "type": "loop_condition",
                        "line": getattr(sub, "lineno", 1),
                        "condition": ast.dump(sub.test, include_attributes=False),
                    }
                )
            elif isinstance(sub, ast.Compare):
                expression = _safe_unparse(sub)
                constraints.append(
                    {
                        "type": "comparison",
                        "line": getattr(sub, "lineno", 1),
                        "expression": expression,
                    }
                )
        return constraints

    def _infer_tainted_variables(self, func_ast: ast.AST) -> Set[str]:
        tainted: Set[str] = set()

        if isinstance(func_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in func_ast.args.args:
                tainted.add(argument.arg)

        changed = True
        while changed:
            changed = False
            for node in ast.walk(func_ast):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = self._assignment_targets(node)
                    value = node.value if isinstance(node, ast.AnnAssign) else node.value
                    if self._expr_is_tainted(value, tainted):
                        for target in targets:
                            if target not in tainted:
                                tainted.add(target)
                                changed = True
        return tainted

    def _assignment_targets(self, node: ast.AST) -> List[str]:
        raw_targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            raw_targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            raw_targets.append(node.target)

        resolved: List[str] = []
        for target in raw_targets:
            resolved.extend(self._extract_name_targets(target))
        return resolved

    def _extract_name_targets(self, node: ast.AST) -> List[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            names: List[str] = []
            for element in node.elts:
                names.extend(self._extract_name_targets(element))
            return names
        return []

    def _expr_is_tainted(self, node: Optional[ast.AST], tainted_variables: Set[str]) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name):
            return node.id in tainted_variables
        if isinstance(node, ast.Call):
            func_name = self._resolve_func_name(node.func)
            if func_name in self.taint_sources:
                return True
            if func_name.split(".")[-1] in self.validation_functions:
                return False
            return any(self._expr_is_tainted(argument, tainted_variables) for argument in node.args)
        if isinstance(node, ast.Attribute):
            attribute_name = self._resolve_func_name(node)
            return attribute_name in self.taint_sources or self._expr_is_tainted(node.value, tainted_variables)
        if isinstance(node, ast.Subscript):
            return self._expr_is_tainted(node.value, tainted_variables)
        if isinstance(node, ast.BinOp):
            return self._expr_is_tainted(node.left, tainted_variables) or self._expr_is_tainted(node.right, tainted_variables)
        if isinstance(node, ast.JoinedStr):
            return any(self._expr_is_tainted(value, tainted_variables) for value in node.values)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(self._expr_is_tainted(element, tainted_variables) for element in node.elts)
        if isinstance(node, ast.Dict):
            return any(self._expr_is_tainted(value, tainted_variables) for value in node.values)
        return False

    def _call_uses_tainted_arguments(self, node: ast.Call, tainted_variables: Set[str]) -> bool:
        return any(self._expr_is_tainted(argument, tainted_variables) for argument in node.args)

    def _is_dynamic_sql(self, node: ast.Call) -> bool:
        if not node.args:
            return False
        query_arg = node.args[0]
        return isinstance(query_arg, (ast.BinOp, ast.JoinedStr, ast.Call))

    def _uses_safe_loader(self, node: ast.Call) -> bool:
        for keyword in node.keywords:
            if keyword.arg == "Loader":
                loader_text = _safe_unparse(keyword.value)
                if "SafeLoader" in loader_text:
                    return True
        return False

    def _resolve_func_name(self, func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            prefix = self._resolve_func_name(func.value)
            return f"{prefix}.{func.attr}" if prefix else func.attr
        if isinstance(func, ast.Call):
            return self._resolve_func_name(func.func)
        return "unknown"

    def _issue(
        self,
        filepath: str,
        node: ast.AST,
        issue_type: str,
        rule_id: str,
        description: str,
        remediation: str,
        severity: float,
        estimated_fix_time: float = 1.0,
        symbol: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AnalysisIssue:
        return AnalysisIssue(
            type=issue_type,
            severity=severity,
            file_path=filepath,
            line=int(getattr(node, "lineno", 1)),
            column=int(getattr(node, "col_offset", 0)),
            category="security",
            description=description,
            rule_id=rule_id,
            estimated_fix_time=estimated_fix_time,
            symbol=symbol,
            remediation=remediation,
            metadata=dict(metadata or {}),
        )


# ---------------------------------------------------------------------------
# Data-flow and interprocedural analysis
# ---------------------------------------------------------------------------
class DataFlowAnalyzer:
    """Call-graph and lightweight interprocedural data-flow analyzer."""

    def __init__(
        self,
        codebase_path: str,
        ast_analyzer: ASTAnalyzer,
        max_call_depth: int,
        analyze_taint: bool,
    ) -> None:
        self.codebase_path = Path(codebase_path).resolve()
        self.ast_analyzer = ast_analyzer
        self.max_call_depth = int(max_call_depth)
        self.analyze_taint = bool(analyze_taint)
        self.call_graph = nx.DiGraph()
        self.data_dependencies = nx.MultiDiGraph()
        self.def_use_chains: Dict[str, Dict[str, List[str]]] = {}
        self.sink_events: List[Dict[str, Any]] = []

    def build_call_graph(self) -> nx.DiGraph:
        self.call_graph.clear()
        for filepath in self._discover_code_files():
            parse_result = self.ast_analyzer.parse_file(filepath)
            if parse_result.tree is None:
                continue
            visitor = DataFlowVisitor(filepath=filepath, analyze_taint=self.analyze_taint)
            visitor.visit(parse_result.tree)
            self.call_graph = nx.compose(self.call_graph, visitor.call_graph)
        return self.call_graph

    def track_data_flow(self) -> None:
        self.data_dependencies.clear()
        self.def_use_chains.clear()
        self.sink_events = []

        for filepath in self._discover_code_files():
            parse_result = self.ast_analyzer.parse_file(filepath)
            if parse_result.tree is None:
                continue
            visitor = DataFlowVisitor(filepath=filepath, analyze_taint=self.analyze_taint)
            visitor.visit(parse_result.tree)
            self.data_dependencies = nx.compose(self.data_dependencies, visitor.dependencies)
            self.def_use_chains.update(visitor.def_use_chains)
            self.sink_events.extend(visitor.sink_events)
            self.call_graph = nx.compose(self.call_graph, visitor.call_graph)

    def detect_interprocedural_issues(self) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []
        for event in self.sink_events:
            if not event.get("tainted", False):
                continue
            issues.append(
                AnalysisIssue(
                    type="security_risk",
                    severity=0.85,
                    file_path=str(event.get("file_path", self.codebase_path)),
                    line=int(event.get("line", 1)),
                    category="security",
                    description="Tainted input reaches a dangerous sink through data flow.",
                    rule_id="SA-DF-001",
                    estimated_fix_time=2.0,
                    symbol=str(event.get("function", "")) or None,
                    remediation="Introduce validation or sanitization before data reaches the sink.",
                    metadata={
                        "sink": event.get("sink"),
                        "tainted_variables": list(event.get("tainted_variables", [])),
                    },
                )
            )

        max_depth = self._estimate_call_depth()
        if max_depth > self.max_call_depth:
            issues.append(
                AnalysisIssue(
                    type="nested_control",
                    severity=_scaled_severity(max_depth, self.max_call_depth),
                    file_path=str(self.codebase_path),
                    line=1,
                    category="quality",
                    description=f"Estimated call depth exceeds configured limit ({max_depth} > {self.max_call_depth}).",
                    rule_id="SA-DF-002",
                    estimated_fix_time=3.0,
                    remediation="Reduce interprocedural depth or simplify recursive call chains.",
                    metadata={"estimated_call_depth": max_depth},
                )
            )
        return issues

    def get_call_graph_summary(self, graph: Optional[nx.DiGraph] = None) -> Dict[str, Any]:
        active_graph = graph if graph is not None else self.call_graph
        nodes = list(active_graph.nodes)
        edges = list(active_graph.edges)
        return {
            "node_count": active_graph.number_of_nodes(),
            "edge_count": active_graph.number_of_edges(),
            "roots": sorted(node for node in nodes if active_graph.in_degree(node) == 0)[:25],
            "leaf_nodes": sorted(node for node in nodes if active_graph.out_degree(node) == 0)[:25],
            "cycles": [list(cycle) for cycle in list(nx.simple_cycles(active_graph))[:10]],
            "estimated_max_depth": self._estimate_call_depth(active_graph),
            "sample_edges": edges[:50],
        }

    def get_data_flow_summary(self) -> Dict[str, Any]:
        return {
            "dependency_node_count": self.data_dependencies.number_of_nodes(),
            "dependency_edge_count": self.data_dependencies.number_of_edges(),
            "tracked_functions": sorted(self.def_use_chains.keys())[:100],
            "sink_event_count": len(self.sink_events),
            "sample_sink_events": self.sink_events[:25],
        }

    def _discover_code_files(self) -> List[str]:
        python_files: List[str] = []
        for root, _, files in os.walk(self.codebase_path):
            for file_name in files:
                if file_name.endswith(".py"):
                    python_files.append(str(Path(root) / file_name))
        return sorted(python_files)

    def _estimate_call_depth(self, graph: Optional[nx.DiGraph] = None) -> int:
        active_graph = graph if graph is not None else self.call_graph
        if active_graph.number_of_nodes() == 0:
            return 0

        condensation = nx.condensation(active_graph)
        if condensation.number_of_nodes() == 0:
            return 0
        if condensation.number_of_edges() == 0:
            return 1

        topo = list(nx.topological_sort(condensation))
        distances = {node: 1 for node in topo}
        for node in topo:
            for successor in condensation.successors(node):
                distances[successor] = max(distances.get(successor, 1), distances[node] + 1)
        return max(distances.values())


class DataFlowVisitor(ast.NodeVisitor):
    """AST visitor that builds call/data-flow structures for a module."""

    DANGEROUS_SINKS = {
        "eval",
        "exec",
        "os.system",
        "subprocess.call",
        "subprocess.run",
        "subprocess.Popen",
        "open",
    }
    TAINT_SOURCES = {
        "input",
        "sys.argv",
        "os.getenv",
        "request.args.get",
        "request.form.get",
        "request.get_json",
    }
    VALIDATORS = {"sanitize", "validate", "escape", "quote", "clean"}

    def __init__(self, filepath: str, analyze_taint: bool = True) -> None:
        self.filepath = filepath
        self.analyze_taint = analyze_taint
        self.current_function: Optional[str] = None
        self.current_tainted_variables: Set[str] = set()
        self.call_graph = nx.DiGraph()
        self.dependencies = nx.MultiDiGraph()
        self.def_use_chains: Dict[str, Dict[str, List[str]]] = {}
        self.sink_events: List[Dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self._visit_callable(node)

    def _visit_callable(self, node: ast.AST) -> None:
        previous_function = self.current_function
        previous_taint = set(self.current_tainted_variables)

        function_name = getattr(node, "name", "<anonymous>")
        qualified_name = self._qualify_function_name(function_name)
        self.current_function = qualified_name
        self.call_graph.add_node(qualified_name)

        params = [argument.arg for argument in getattr(node.args, "args", [])]
        self.current_tainted_variables = set(params) if self.analyze_taint else set()
        self.def_use_chains.setdefault(qualified_name, {"defs": list(params), "uses": []})

        self.generic_visit(node)

        self.current_function = previous_function
        self.current_tainted_variables = previous_taint

    def visit_Assign(self, node: ast.Assign) -> Any:
        self._record_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._record_assignment([node.target], node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        callee = self._resolve_callee(node.func)
        if self.current_function:
            self.call_graph.add_edge(self.current_function, callee)
            for index, argument in enumerate(node.args):
                variable_names = sorted(self._extract_variables(argument))
                tainted = any(name in self.current_tainted_variables for name in variable_names)
                self.dependencies.add_edge(
                    self.current_function,
                    callee,
                    arg_index=index,
                    var_names=variable_names,
                    tainted=tainted,
                )
                self.def_use_chains.setdefault(self.current_function, {"defs": [], "uses": []})
                self.def_use_chains[self.current_function]["uses"].extend(variable_names)

            if self._is_validator_call(callee):
                for argument in node.args:
                    self.current_tainted_variables.difference_update(self._extract_variables(argument))

            if callee in self.DANGEROUS_SINKS:
                tainted_variables = sorted(
                    {name for argument in node.args for name in self._extract_variables(argument) if name in self.current_tainted_variables}
                )
                self.sink_events.append(
                    {
                        "file_path": self.filepath,
                        "line": getattr(node, "lineno", 1),
                        "function": self.current_function,
                        "sink": callee,
                        "tainted": bool(tainted_variables),
                        "tainted_variables": tainted_variables,
                    }
                )

        self.generic_visit(node)

    def _record_assignment(self, targets: Sequence[ast.AST], value: Optional[ast.AST]) -> None:
        if not self.current_function:
            return

        target_names: Set[str] = set()
        for target in targets:
            target_names.update(self._extract_variables(target))

        self.def_use_chains.setdefault(self.current_function, {"defs": [], "uses": []})
        self.def_use_chains[self.current_function]["defs"].extend(sorted(target_names))

        if self.analyze_taint and value is not None and self._expr_is_tainted(value):
            self.current_tainted_variables.update(target_names)

    def _expr_is_tainted(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.current_tainted_variables
        if isinstance(node, ast.Call):
            callee = self._resolve_callee(node.func)
            if callee in self.TAINT_SOURCES:
                return True
            if self._is_validator_call(callee):
                return False
            return any(self._expr_is_tainted(argument) for argument in node.args)
        if isinstance(node, ast.Attribute):
            return self._resolve_callee(node) in self.TAINT_SOURCES or self._expr_is_tainted(node.value)
        if isinstance(node, ast.Subscript):
            return self._expr_is_tainted(node.value)
        if isinstance(node, ast.BinOp):
            return self._expr_is_tainted(node.left) or self._expr_is_tainted(node.right)
        if isinstance(node, ast.JoinedStr):
            return any(self._expr_is_tainted(value) for value in node.values)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(self._expr_is_tainted(element) for element in node.elts)
        if isinstance(node, ast.Dict):
            return any(self._expr_is_tainted(value) for value in node.values)
        return False

    def _qualify_function_name(self, function_name: str) -> str:
        relative_path = Path(self.filepath).name
        return f"{relative_path}:{function_name}"

    def _resolve_callee(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._resolve_callee(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Call):
            return self._resolve_callee(node.func)
        return "unknown"

    def _extract_variables(self, node: ast.AST) -> Set[str]:
        variables: Set[str] = set()
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name):
                variables.add(subnode.id)
        return variables

    def _is_validator_call(self, callee: str) -> bool:
        leaf_name = callee.split(".")[-1]
        return leaf_name in self.VALIDATORS


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


def _clamp_severity(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _scaled_severity(observed: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    ratio = observed / threshold
    return max(0.4, min(1.0, 0.5 + (ratio - 1.0) * 0.5))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print("\n=== Running Static Analyzer ===\n")
    import json

    codebase_path = "src/agents/evaluators/"
    analyzer = StaticAnalyzer(codebase_path=codebase_path)
    report = analyzer.full_analysis()

    logger.info("Static analysis report:\n%s", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\n=== Successfully Ran Static Analyzer ===\n")
