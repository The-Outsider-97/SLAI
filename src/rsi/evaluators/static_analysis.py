import os
import tempfile
import subprocess
import logging
import json
from collections import defaultdict
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Analysis severity levels
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def static_analysis_bandit(code: str) -> Dict[str, Any]:
    """
    Perform static code analysis using Bandit, Pylint, and Mypy.
    Returns a consolidated report with severity levels and issues.
    
    Parameters:
    - code (str): Source code to analyze.

    Returns:
    - Dict: Structured analysis report including severity, issues, and scores.
    """

    logger.info("Starting comprehensive static analysis...")

    with tempfile.TemporaryDirectory() as tmpdirname:
        code_file = os.path.join(tmpdirname, "static_analysis_module.py")

        # Save the generated code to a temp file
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        logger.debug(f"Code saved to temporary path {code_file}")

        # Run Bandit Security Analysis
        bandit_results = run_bandit(code_file)

        # Run Pylint Code Quality Analysis
        pylint_results = run_pylint(code_file)

        # Run Mypy Type Checking
        mypy_results = run_mypy(code_file)

        # Consolidate all results
        final_report = consolidate_analysis(bandit_results, pylint_results, mypy_results)

        # Risk assessment and severity assignment
        risk_level = assess_risk(final_report)

        logger.info(f"Static analysis completed. Risk level: {risk_level}")
        logger.debug(f"Full analysis report:\n{json.dumps(final_report, indent=2)}")

        return {
            "risk_level": risk_level,
            "report": final_report
        }


def run_bandit(code_file: str) -> Dict[str, Any]:
    """
    Runs Bandit static security analysis.
    Returns a dictionary of issues and severity.
    """

    logger.debug("Running Bandit security analysis...")
    bandit_cmd = ["bandit", "-r", code_file, "-f", "json"]

    try:
        result = subprocess.run(bandit_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        bandit_output = json.loads(result.stdout.decode())

        issues = []
        for issue in bandit_output.get("results", []):
            issues.append({
                "line_number": issue.get("line_number"),
                "issue_severity": issue.get("issue_severity"),
                "issue_confidence": issue.get("issue_confidence"),
                "issue_text": issue.get("issue_text"),
                "code": issue.get("code")
            })

        logger.debug(f"Bandit found {len(issues)} issues.")
        return {"tool": "bandit", "issues": issues}

    except subprocess.TimeoutExpired:
        logger.warning("Bandit scan timed out.")
        return {"tool": "bandit", "issues": [{"issue_severity": "CRITICAL", "issue_text": "Bandit scan timeout."}]}

    except Exception as e:
        logger.error(f"Bandit scan failed: {e}")
        return {"tool": "bandit", "issues": [{"issue_severity": "CRITICAL", "issue_text": str(e)}]}


def run_pylint(code_file: str) -> Dict[str, Any]:
    """
    Runs Pylint for code quality and style.
    Returns a dictionary of issues with their severity.
    """

    logger.debug("Running Pylint code quality analysis...")
    pylint_cmd = ["pylint", code_file, "-f", "json"]

    try:
        result = subprocess.run(pylint_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        pylint_output = json.loads(result.stdout.decode())

        issues = []
        for message in pylint_output:
            severity = map_pylint_type_to_severity(message.get("type", "convention"))
            issues.append({
                "line": message.get("line"),
                "symbol": message.get("symbol"),
                "message": message.get("message"),
                "severity": severity
            })

        logger.debug(f"Pylint found {len(issues)} issues.")
        return {"tool": "pylint", "issues": issues}

    except subprocess.TimeoutExpired:
        logger.warning("Pylint scan timed out.")
        return {"tool": "pylint", "issues": [{"severity": "CRITICAL", "message": "Pylint scan timeout."}]}

    except Exception as e:
        logger.error(f"Pylint scan failed: {e}")
        return {"tool": "pylint", "issues": [{"severity": "CRITICAL", "message": str(e)}]}


def run_mypy(code_file: str) -> Dict[str, Any]:
    """
    Runs Mypy for static type checking.
    Returns a dictionary of issues.
    """

    logger.debug("Running Mypy type checking...")
    mypy_cmd = ["mypy", code_file, "--ignore-missing-imports", "--no-color-output", "--no-error-summary"]

    try:
        result = subprocess.run(mypy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        output = result.stdout.decode()

        issues = []
        for line in output.splitlines():
            if line.strip() and ": error:" in line:
                parts = line.split(":")
                issues.append({
                    "file": parts[0].strip(),
                    "line": int(parts[1].strip()),
                    "column": int(parts[2].strip()),
                    "message": ":".join(parts[3:]).strip(),
                    "severity": "MEDIUM"
                })

        logger.debug(f"Mypy found {len(issues)} type issues.")
        return {"tool": "mypy", "issues": issues}

    except subprocess.TimeoutExpired:
        logger.warning("Mypy scan timed out.")
        return {"tool": "mypy", "issues": [{"severity": "CRITICAL", "message": "Mypy scan timeout."}]}

    except Exception as e:
        logger.error(f"Mypy scan failed: {e}")
        return {"tool": "mypy", "issues": [{"severity": "CRITICAL", "message": str(e)}]}


def consolidate_analysis(*tool_results) -> Dict[str, Any]:
    """
    Consolidates analysis results from Bandit, Pylint, and Mypy into a unified report.
    """

    logger.debug("Consolidating analysis results...")
    consolidated_report = defaultdict(list)

    for tool_result in tool_results:
        tool_name = tool_result.get("tool")
        issues = tool_result.get("issues", [])
        consolidated_report[tool_name].extend(issues)

    return consolidated_report


def assess_risk(report: Dict[str, Any]) -> str:
    """
    Determines the risk level based on the severity of issues found in the report.
    
    Returns:
    - str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
    """

    logger.debug("Assessing risk from consolidated report...")

    severity_counts = defaultdict(int)

    # Count issues per severity level
    for tool_issues in report.values():
        for issue in tool_issues:
            severity = issue.get("issue_severity") or issue.get("severity") or "LOW"
            severity = severity.upper()
            if severity in SEVERITY_LEVELS:
                severity_counts[severity] += 1

    logger.debug(f"Severity counts: {dict(severity_counts)}")

    # Determine overall risk based on counts
    if severity_counts["CRITICAL"] > 0:
        return "CRITICAL"
    elif severity_counts["HIGH"] > 0:
        return "HIGH"
    elif severity_counts["MEDIUM"] > 0:
        return "MEDIUM"
    return "LOW"


def map_pylint_type_to_severity(pylint_type: str) -> str:
    """
    Maps pylint message types to severity levels.
    """

    mapping = {
        "error": "HIGH",
        "warning": "MEDIUM",
        "refactor": "MEDIUM",
        "convention": "LOW",
        "info": "LOW"
    }

    return mapping.get(pylint_type.lower(), "LOW")
