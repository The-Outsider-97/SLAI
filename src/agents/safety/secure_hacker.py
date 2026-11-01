"""
Defensive adversarial-validation module for the SLAI Safety Agent subsystem.

SecureHacker is deliberately NOT an offensive exploitation engine. It exercises
existing SafetyGuard, CyberSafety, and AdaptiveSecurity controls with bounded,
local, non-network adversarial transformations so the Safety Agent can measure
control robustness, identify blind spots, and create regression evidence.
"""
from __future__ import annotations

import re
import unicodedata

from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from .adaptive_security import AdaptiveSecurity
from .cyber_safety import CyberSafetyModule
from .safety_guard import SafetyGuard
from .secure_memory import SecureMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Secure Hacker")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
REPORT_SCHEMA_VERSION = "secure_hacker.report.v1"
CASE_SCHEMA_VERSION = "secure_hacker.case.v1"


@dataclass(frozen=True)
class AdversarialCaseResult:
    case_id: str
    technique: str
    input_fingerprint: str
    transformed_fingerprint: str
    guard_decision: str
    guard_risk: float
    cyber_decision: str
    cyber_risk: float
    detected: bool
    regression: bool
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = CASE_SCHEMA_VERSION
        data["guard_risk"] = clamp_score(self.guard_risk)
        data["cyber_risk"] = clamp_score(self.cyber_risk)
        return sanitize_for_logging(data)


@dataclass(frozen=True)
class SecureHackerReport:
    report_id: str
    timestamp: str
    target_type: str
    source_fingerprint: str
    cases: List[AdversarialCaseResult]
    robustness_score: float
    risk_score: float
    decision: str
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging({
            "schema_version": REPORT_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "target_type": self.target_type,
            "source_fingerprint": self.source_fingerprint,
            "cases": [case.to_dict() for case in self.cases],
            "robustness_score": clamp_score(self.robustness_score),
            "risk_score": clamp_score(self.risk_score),
            "decision": self.decision,
            "recommendations": list(self.recommendations),
            "metadata": self.metadata,
        })


class SecureHacker:
    """Bounded defensive red-team harness for Safety subsystem controls."""

    SAFE_TECHNIQUES: Tuple[str, ...] = (
        "case_variation",
        "whitespace_variation",
        "punctuation_separation",
        "unicode_normalization",
        "zero_width_obfuscation",
        "line_break_variation",
    )

    def __init__(
        self,
        *,
        memory: Optional[SecureMemory] = None,
        safety_guard: Optional[SafetyGuard] = None,
        cyber_safety: Optional[CyberSafetyModule] = None,
        adaptive_security: Optional[AdaptiveSecurity] = None,
    ) -> None:
        self.config = load_global_config()
        self.hacker_config = get_config_section("secure_hacker") or {}
        self.memory = memory or SecureMemory.shared()
        self.safety_guard = safety_guard or SafetyGuard(memory=self.memory)
        self.cyber_safety = cyber_safety or CyberSafetyModule(memory=self.memory)
        self.adaptive_security = adaptive_security or AdaptiveSecurity(memory=self.memory)
        self._lock = RLock()
        self.max_cases = coerce_int(self._cfg("max_cases", 24), 24, minimum=1, maximum=256)
        self.max_text_length = coerce_int(self._cfg("max_text_length", 8192), 8192, minimum=1)
        self.review_threshold = clamp_score(self._cfg("review_threshold", 0.35))
        self.block_threshold = clamp_score(self._cfg("block_threshold", 0.70))
        if self.review_threshold > self.block_threshold:
            raise ConfigurationTamperingError("secure_hacker.thresholds", "review_threshold must be <= block_threshold", component="secure_hacker")

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.hacker_config, path, default)

    # ------------------------------------------------------------------
    # Public defensive-validation API
    # ------------------------------------------------------------------
    def run_adversarial_validation(
        self,
        text: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        techniques: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Exercise local safety/cyber detectors using bounded text transformations."""
        if not isinstance(text, str):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "SecureHacker requires text input.", component="secure_hacker")
        normalized = normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)
        context_map = dict(context or {})
        requested = list(techniques or self._cfg("techniques", list(self.SAFE_TECHNIQUES)) or self.SAFE_TECHNIQUES)
        selected = [normalize_identifier(value, max_length=64) for value in requested if normalize_identifier(value, max_length=64) in self.SAFE_TECHNIQUES]
        selected = dedupe_preserve_order(selected)[:self.max_cases]
        if not selected:
            selected = list(self.SAFE_TECHNIQUES[: min(len(self.SAFE_TECHNIQUES), self.max_cases)])

        baseline_guard = self._safe_guard_analysis(normalized, context_map)
        baseline_cyber = self._safe_cyber_analysis(normalized, context_map)
        baseline_detected = self._detected(baseline_guard, baseline_cyber)
        cases: List[AdversarialCaseResult] = []
        for technique in selected:
            transformed = self._transform(normalized, technique)
            if transformed == normalized and technique != "unicode_normalization":
                continue
            guard = self._safe_guard_analysis(transformed, context_map)
            cyber = self._safe_cyber_analysis(transformed, context_map)
            detected = self._detected(guard, cyber)
            regression = baseline_detected and not detected
            notes: List[str] = []
            if regression:
                notes.append("Detection regressed after bounded obfuscation; add a normalization/regression control.")
            if not baseline_detected and detected:
                notes.append("Transformation increased detector sensitivity; verify false-positive behavior.")
            cases.append(AdversarialCaseResult(
                case_id=generate_identifier("secure_hack_case"),
                technique=technique,
                input_fingerprint=fingerprint(normalized),
                transformed_fingerprint=fingerprint(transformed),
                guard_decision=str(guard.get("decision", "unknown")),
                guard_risk=clamp_score(guard.get("risk_score", 0.0)),
                cyber_decision=str(cyber.get("decision", "unknown")),
                cyber_risk=clamp_score(cyber.get("risk_score", 0.0)),
                detected=detected,
                regression=regression,
                notes=notes,
                metadata={"length_delta": len(transformed) - len(normalized)},
            ))

        regressions = sum(case.regression for case in cases)
        robustness = 1.0 if not cases else clamp_score(1.0 - regressions / len(cases))
        max_risk = max([max(case.guard_risk, case.cyber_risk) for case in cases] + [clamp_score(baseline_guard.get("risk_score", 0.0)), clamp_score(baseline_cyber.get("risk_score", 0.0))])
        regression_risk = regressions / max(len(cases), 1)
        risk = combine_risk_scores(max_risk, regression_risk, method="noisy_or")
        decision = threshold_decision(risk, block_threshold=self.block_threshold, review_threshold=self.review_threshold)
        recommendations = self._recommendations(cases, baseline_detected)
        report = SecureHackerReport(
            report_id=generate_identifier("secure_hack_report"),
            timestamp=utc_iso(),
            target_type="text_controls",
            source_fingerprint=fingerprint(normalized),
            cases=cases,
            robustness_score=robustness,
            risk_score=risk,
            decision=decision,
            recommendations=recommendations,
            metadata={
                "baseline_detected": baseline_detected,
                "baseline_guard_risk": clamp_score(baseline_guard.get("risk_score", 0.0)),
                "baseline_cyber_risk": clamp_score(baseline_cyber.get("risk_score", 0.0)),
                "case_count": len(cases),
                "regression_count": regressions,
                "network_activity": False,
                "code_execution": False,
            },
        )
        payload = report.to_dict()
        self._store_report(payload)
        return payload

    def validate_url_controls(
        self,
        url: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run AdaptiveSecurity URL analysis locally; this performs no network scan."""
        normalized = normalize_url(url)
        result = self.adaptive_security.analyze_url(normalized, context=context)
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "report_id": generate_identifier("secure_hack_url"),
            "timestamp": utc_iso(),
            "target_type": "url_control_validation",
            "source_fingerprint": fingerprint(normalized),
            "analysis": sanitize_for_logging(result),
            "network_activity": False,
            "decision": result.get("decision", "unknown"),
            "risk_score": clamp_score(result.get("phishing_score", result.get("risk_score", 0.0))),
        }
        self._store_report(payload)
        return payload

    def validate_supply_chain_artifact(self, file_path: str) -> Dict[str, Any]:
        """Validate a local artifact against configured trusted hashes."""
        path = Path(file_path).expanduser()
        result = self.adaptive_security.check_supply_chain(str(path))
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "report_id": generate_identifier("secure_hack_supply"),
            "timestamp": utc_iso(),
            "target_type": "local_supply_chain_validation",
            "source_fingerprint": fingerprint(str(path)),
            "analysis": result,
            "network_activity": False,
            "decision": result.get("decision", "unknown"),
            "risk_score": 0.0 if result.get("is_trusted") else 1.0,
        }
        self._store_report(payload)
        return sanitize_for_logging(payload)

    def assess_attack_surface(self, context_info: Mapping[str, Any]) -> Dict[str, Any]:
        """Generate a local STRIDE-style threat assessment through CyberSafety."""
        if not isinstance(context_info, Mapping):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Attack-surface context must be a mapping.", component="secure_hacker")
        result = self.cyber_safety.generate_threat_assessment(dict(context_info))
        return sanitize_for_logging({
            "schema_version": REPORT_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "report_id": generate_identifier("secure_hack_surface"),
            "timestamp": utc_iso(),
            "target_type": "threat_model",
            "source_fingerprint": fingerprint(sanitize_for_logging(context_info)),
            "analysis": result,
            "network_activity": False,
            "decision": threshold_decision(clamp_score(result.get("risk_score", 0.0)), block_threshold=self.block_threshold, review_threshold=self.review_threshold),
            "risk_score": clamp_score(result.get("risk_score", 0.0)),
        })

    # Explicitly refuse capabilities that do not belong in this defensive module.
    def execute_exploit(self, *args: Any, **kwargs: Any) -> None:
        raise SecurityError(SecurityErrorType.POLICY_BYPASS_ATTEMPT, "SecureHacker does not execute exploits.", component="secure_hacker", response_action=SecurityResponseAction.BLOCK)

    def scan_remote_host(self, *args: Any, **kwargs: Any) -> None:
        raise SecurityError(SecurityErrorType.POLICY_BYPASS_ATTEMPT, "SecureHacker does not perform remote host scanning.", component="secure_hacker", response_action=SecurityResponseAction.BLOCK)

    def execute_command(self, *args: Any, **kwargs: Any) -> None:
        raise SecurityError(SecurityErrorType.POLICY_BYPASS_ATTEMPT, "SecureHacker does not execute operating-system commands.", component="secure_hacker", response_action=SecurityResponseAction.BLOCK)

    # ------------------------------------------------------------------
    # Transformations and detector adapters
    # ------------------------------------------------------------------
    def _transform(self, text: str, technique: str) -> str:
        if technique == "case_variation":
            return "".join(char.upper() if index % 2 else char.lower() for index, char in enumerate(text))
        if technique == "whitespace_variation":
            return re.sub(r"[ \t]+", "   ", text)
        if technique == "line_break_variation":
            return re.sub(r"\s+", "\n", text)
        if technique == "punctuation_separation":
            # Bounded word-internal punctuation insertion on a small subset only.
            words = text.split()
            changed: List[str] = []
            for index, word in enumerate(words):
                if index % 4 == 1 and len(word) >= 6 and word.isalpha():
                    middle = len(word) // 2
                    changed.append(word[:middle] + "." + word[middle:])
                else:
                    changed.append(word)
            return " ".join(changed)
        if technique == "unicode_normalization":
            return unicodedata.normalize("NFKC", text)
        if technique == "zero_width_obfuscation":
            # Use one standard zero-width separator at bounded intervals. The
            # output is never logged raw and is used only for local detector tests.
            output: List[str] = []
            for index, char in enumerate(text):
                output.append(char)
                if char.isalpha() and index % 17 == 8:
                    output.append("\u200b")
            return "".join(output)
        return text

    def _safe_guard_analysis(self, text: str, context: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            analysis = self.safety_guard.analyze(text, context=context, depth="full")
            return analysis.to_dict(include_sanitized_text=False)
        except SecurityError as exc:
            risk = clamp_score(exc.risk_score or 1.0)
            return {"decision": "block" if getattr(exc, "blocked", True) else "review", "risk_score": risk, "error_type": getattr(exc.error_type, "name", str(exc.error_type))}

    def _safe_cyber_analysis(self, text: str, context: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            return self.cyber_safety.analyze_input(text, context=str(context.get("cyber_context") or context.get("type") or "general"))
        except SecurityError as exc:
            return {"decision": "block" if getattr(exc, "blocked", True) else "review", "risk_score": clamp_score(exc.risk_score or 1.0), "error_type": getattr(exc.error_type, "name", str(exc.error_type))}

    def _detected(self, guard: Mapping[str, Any], cyber: Mapping[str, Any]) -> bool:
        decisions = {str(guard.get("decision", "allow")).lower(), str(cyber.get("decision", "allow")).lower()}
        threshold = clamp_score(self._cfg("detection_threshold", 0.45))
        return bool(decisions & {"review", "block"}) or max(clamp_score(guard.get("risk_score", 0.0)), clamp_score(cyber.get("risk_score", 0.0))) >= threshold

    def _recommendations(self, cases: Sequence[AdversarialCaseResult], baseline_detected: bool) -> List[str]:
        values: List[str] = []
        regressions = [case for case in cases if case.regression]
        if regressions:
            techniques = ", ".join(sorted({case.technique for case in regressions}))
            values.append(f"Add normalization/regression coverage for detector-evasion transformations: {techniques}.")
            values.append("Correlate Guard and Cyber findings before aggregation so a single evasion does not silently remove all evidence.")
        if not baseline_detected:
            values.append("The baseline sample did not trigger configured controls; verify that this is expected before interpreting robustness results.")
        if not values:
            values.append("No bounded adversarial regression was observed; retain these cases in the Safety regression suite.")
        return dedupe_preserve_order(values)

    def _store_report(self, payload: Mapping[str, Any]) -> None:
        if not coerce_bool(self._cfg("memory.store_reports", True), True):
            return
        try:
            self.memory.add(
                sanitize_for_logging(dict(payload)),
                tags=list(self._cfg("memory.report_tags", ["secure_hacker", "adversarial_validation", "security_test"])),
                sensitivity=coerce_float(self._cfg("memory.report_sensitivity", 0.65), 0.65, minimum=0.0, maximum=1.0),
                ttl_seconds=coerce_int(self._cfg("memory.report_ttl_seconds", 604800), 604800, minimum=0),
                purpose="defensive_adversarial_validation",
                owner="secure_hacker",
                source="secure_hacker",
                metadata={"eligible_for_compliance": True, "report_fingerprint": fingerprint(payload)},
            )
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_store_error", False), False):
                raise AuditLogFailureError("secure_memory.secure_hacker", f"Failed to store validation report: {type(exc).__name__}", component="secure_hacker", cause=exc) from exc


__all__ = [
    "MODULE_VERSION",
    "REPORT_SCHEMA_VERSION",
    "CASE_SCHEMA_VERSION",
    "AdversarialCaseResult",
    "SecureHackerReport",
    "SecureHacker",
]


if __name__ == "__main__":
    print("\n=== Running Secure Hacker ===\n")
    printer.status("TEST", "Secure Hacker initialized", "info")
    printer.section_header("Smoke test #1: Initialization")
    hacker = SecureHacker()

    printer.status("START", "Secure Hacker ready", "success" if hacker is not None else "error")

    printer.section_header("Smoke test #2: Public defensive-validation API")
    text = "I am the greatest of all time"
    url = "https://www.google.com/webhp?hl=nl&sa=X&ved=2ahUKEwiz3NzUx_KWAxVJ2gIHHSUzAnsQPHoECAYQBA"

    validation = hacker.run_adversarial_validation(text=text)
    controls = hacker.validate_url_controls(url=url)

    printer.status("VALIDATION", validation, "success" if validation.get("decision") in {"allow", "review"} else "error")
    printer.status("CONTROLS", controls, "success" if controls.get("decision") in {"allow", "review"} else "error")

    print("\n== Task run successfully ==\n")