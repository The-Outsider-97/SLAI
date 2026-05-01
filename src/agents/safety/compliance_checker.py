"""
Production-grade compliance checker for the Safety Agent subsystem.

This module evaluates security, privacy, model-governance, and operational
controls against a YAML/file-backed compliance framework and runtime evidence
stored in SecureMemory. It is a compliance orchestration layer only: it does not
own feature extraction, model training, secure memory persistence, incident
response, or UI rendering.

Design goals:
- keep compliance control definitions configuration-backed and auditable;
- avoid leaking raw PII, secrets, model payloads, prompts, or identifiers in
  logs, memory entries, reports, or exceptions;
- preserve backwards-compatible public methods used by the current subsystem;
- use shared safety helpers for redaction, hashing, serialization, scoring,
  timestamps, and safe log payloads instead of re-implementing them locally;
- use structured security errors for configuration, integrity, access, audit,
  and unsafe-evaluation failures;
- fail closed when required compliance evidence or framework definitions are
  missing, malformed, or tampered with.
"""

from __future__ import annotations

import json
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.safety_helpers import *
from .utils.security_error import *
from .secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Security Compliance Checker")
printer = PrettyPrinter()

MODULE_VERSION = "2.1.0"
EVALUATION_SCHEMA_VERSION = "compliance_checker.evaluation.v3"
REPORT_SCHEMA_VERSION = "compliance_checker.report.v2"


@dataclass(frozen=True)
class EvidenceItem:
    """Audit-safe evidence item for a compliance control."""

    source: str
    summary: str
    status: str = "observed"
    fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging(asdict(self))


@dataclass(frozen=True)
class ControlEvaluation:
    """Structured evaluation for one compliance control."""

    control_id: str
    objective: str
    status: str
    score: float
    severity: str
    owner: str
    section_id: str
    section_title: str
    evaluator: str
    findings: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    remediation: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = clamp_score(data["score"])
        data["evidence"] = [item.to_dict() if isinstance(item, EvidenceItem) else sanitize_for_logging(item) for item in self.evidence]
        return sanitize_for_logging(data)


@dataclass(frozen=True)
class SectionEvaluation:
    """Structured evaluation for a compliance framework section."""

    section_id: str
    title: str
    status: str
    score: float
    weight: float
    controls: List[ControlEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging({
            "section_id": self.section_id,
            "title": self.title,
            "status": self.status,
            "score": clamp_score(self.score),
            "weight": self.weight,
            "results": [control.to_dict() for control in self.controls],
            "control_count": len(self.controls),
            "pass_count": sum(1 for control in self.controls if control.status == "pass"),
            "warning_count": sum(1 for control in self.controls if control.status == "warning"),
            "fail_count": sum(1 for control in self.controls if control.status == "fail"),
            "error_count": sum(1 for control in self.controls if control.status == "error"),
        })


class ComplianceChecker:
    """
    Evaluates configured compliance controls using SecureMemory evidence.

    Backwards-compatible public methods are retained:
    - evaluate_compliance()
    - generate_report(results)
    - check_gdpr(control)
    - check_hipaa(data)
    - _evaluate_control(control)
    - _generic_control_check(control)
    - _check_data_classification(control)
    - _check_data_minimization(control)
    - _check_model_integrity(control)
    """

    def __init__(self):
        self.config = load_global_config()
        self.complience_config = get_config_section("compliance_checker")  # Backwards-compatible misspelling.
        self.compliance_config = self.complience_config
        self.compliance_file_path = self.compliance_config.get("compliance_file_path")
        self.phishing_model_path = self.compliance_config.get("phishing_model_path")
        self.enable_memory_bootstrap = coerce_bool(self.compliance_config.get("enable_memory_bootstrap"), True)
        self.report_thresholds = dict(self.compliance_config.get("report_thresholds", {}))
        self.weights = dict(self.compliance_config.get("weights", {}))
        self._validate_configuration()

        self.memory = SecureMemory()
        if self.enable_memory_bootstrap and hasattr(self.memory, "bootstrap_if_empty"):
            self.memory.bootstrap_if_empty()

        self.control_evaluators: Dict[str, Callable[[Mapping[str, Any], str, str], ControlEvaluation]] = {
            "DP-001": self._evaluate_data_classification,
            "DP-002": self._evaluate_gdpr,
            "DP-003": self._evaluate_data_minimization,
            "MS-001": self._evaluate_model_integrity,
            "AS-001": self._evaluate_application_security,
            "OS-001": self._evaluate_operational_security,
            "AL-001": self._evaluate_audit_logging,
            "PR-001": self._evaluate_privacy_redaction,
        }
        self.compliance_framework = self._load_compliance_framework()
        self.framework_fingerprint = fingerprint(self.compliance_framework)

        logger.info("Compliance Checker initialized: %s", stable_json(safe_log_payload(
            "compliance_checker.initialized",
            {
                "framework_version": get_nested(self.compliance_framework, "documentInfo.version", "unknown"),
                "framework_fingerprint": self.framework_fingerprint,
                "sections": len(self.compliance_framework.get("sections", [])),
            },
        )))

    # ------------------------------------------------------------------
    # Configuration and framework loading
    # ------------------------------------------------------------------
    def _validate_configuration(self) -> None:
        require_keys(
            self.compliance_config,
            ["report_thresholds", "weights"],
            context="compliance_checker",
        )
        thresholds = self.compliance_config.get("report_thresholds", {})
        critical = clamp_score(thresholds.get("critical", 0.0))
        warning = clamp_score(thresholds.get("warning", 0.0))
        if critical <= 0 or warning <= 0 or critical > warning:
            raise ConfigurationTamperingError(
                config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
                suspicious_change="compliance_checker.report_thresholds must define 0 < critical <= warning <= 1",
                component="compliance_checker",
            )
        if not isinstance(self.weights, Mapping) or not self.weights:
            raise ConfigurationTamperingError(
                config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
                suspicious_change="compliance_checker.weights must be a non-empty mapping",
                component="compliance_checker",
            )

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.compliance_config, path, default)

    def _resolve_path(self, raw_path: Optional[Union[str, Path]]) -> Optional[Path]:
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute():
            return path
        config_path = Path(str(self.config.get("__config_path__", ""))).resolve()
        candidates = [
            Path.cwd() / path,
            config_path.parent / path if config_path else path,
            config_path.parent.parent.parent / path if config_path else path,
            Path("/mnt/data") / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_compliance_framework(self) -> Dict[str, Any]:
        """Load framework from SecureMemory, file, or YAML-backed inline config."""

        framework_entries = self._recall_memory("compliance_framework", top_k=1)
        if framework_entries:
            framework = self._entry_data(framework_entries[0])
            if isinstance(framework, Mapping) and framework.get("sections"):
                return dict(framework)

        path = self._resolve_path(self.compliance_file_path)
        if path and path.exists():
            try:
                raw = load_text_file(path, max_bytes=coerce_int(self._cfg("max_framework_bytes", 1_048_576), 1_048_576, minimum=4096))
                framework = parse_json_object(raw, context="compliance_framework")
                self._validate_framework(framework)
                self._store_memory(
                    framework,
                    tags=["compliance_framework", "security", "governance"],
                    sensitivity=coerce_float(self._cfg("memory.framework_sensitivity", 0.8), 0.8, minimum=0.0, maximum=1.0),
                    metadata={"source": str(path), "fingerprint": fingerprint(framework)},
                )
                return framework
            except SecurityError:
                raise
            except Exception as exc:
                raise wrap_security_exception(
                    exc,
                    operation="load_compliance_framework",
                    component="compliance_checker",
                    context={"path": str(path), "path_fingerprint": fingerprint(str(path))},
                    error_type=SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                    severity=SecuritySeverity.HIGH,
                ) from exc

        inline_framework = self.compliance_config.get("framework")
        if isinstance(inline_framework, Mapping) and inline_framework.get("sections"):
            framework = dict(inline_framework)
            self._validate_framework(framework)
            self._store_memory(
                framework,
                tags=["compliance_framework", "security", "governance"],
                sensitivity=coerce_float(self._cfg("memory.framework_sensitivity", 0.8), 0.8, minimum=0.0, maximum=1.0),
                metadata={"source": "secure_config.yaml", "fingerprint": fingerprint(framework)},
            )
            return framework

        raise ConfigurationTamperingError(
            config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
            suspicious_change="No valid compliance framework found in secure memory, compliance_file_path, or compliance_checker.framework",
            component="compliance_checker",
        )

    def _validate_framework(self, framework: Mapping[str, Any]) -> None:
        require_keys(framework, ["documentInfo", "sections"], context="compliance_framework")
        if not isinstance(framework.get("sections"), list) or not framework["sections"]:
            raise ConfigurationTamperingError(
                config_file_path=str(self.compliance_file_path or self.config.get("__config_path__", "secure_config.yaml")),
                suspicious_change="compliance framework must contain at least one section",
                component="compliance_checker",
            )
        seen_controls: set[str] = set()
        for section in framework["sections"]:
            require_keys(section, ["sectionId", "title"], context="compliance_framework.section")
            for control in self._iter_section_controls(section):
                require_keys(control, ["controlId", "objective"], context=f"compliance_framework.section.{section['sectionId']}.control")
                control_id = str(control["controlId"])
                if control_id in seen_controls:
                    raise ConfigurationTamperingError(
                        config_file_path=str(self.compliance_file_path or self.config.get("__config_path__", "secure_config.yaml")),
                        suspicious_change=f"duplicate compliance control ID detected: {control_id}",
                        component="compliance_checker",
                    )
                seen_controls.add(control_id)

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------
    def evaluate_compliance(self) -> Dict[str, Any]:
        """Comprehensively evaluate configured controls and return an audit-safe dict."""

        printer.status("CHECK", "Evaluating compliance", "info")
        started_at = utc_iso()
        sections: Dict[str, Any] = {}
        section_scores: Dict[str, float] = {}
        section_weights: Dict[str, float] = {}
        all_controls: List[ControlEvaluation] = []

        for section in self.compliance_framework.get("sections", []):
            section_id = normalize_identifier(section.get("sectionId"), max_length=96).upper()
            section_title = normalize_text(section.get("title", section_id), max_length=256)
            controls = [self._evaluate_control_record(control, section_id, section_title) for control in self._iter_section_controls(section)]
            section_score = self._score_controls(controls)
            section_status = self._get_compliance_status(section_score)
            section_weight = coerce_float(section.get("weight", self.weights.get(section_id.lower(), self.weights.get(normalize_identifier(section_title), 1.0))), 1.0, minimum=0.0)
            section_eval = SectionEvaluation(section_id, section_title, section_status, section_score, section_weight, controls)
            sections[section_id] = section_eval.to_dict()
            section_scores[section_id] = section_score
            section_weights[section_id] = section_weight
            all_controls.extend(controls)

        overall_score = weighted_average(section_scores, section_weights, default=0.0) if section_scores else 0.0
        status = self._get_compliance_status(overall_score)
        failed_controls = [control for control in all_controls if control.status in {"fail", "error"}]
        warning_controls = [control for control in all_controls if control.status == "warning"]
        results: Dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "timestamp": started_at,
            "completed_at": utc_iso(),
            "framework": {
                "version": get_nested(self.compliance_framework, "documentInfo.version", "unknown"),
                "name": get_nested(self.compliance_framework, "documentInfo.name", get_nested(self.compliance_framework, "documentInfo.title", "Safety Compliance Framework")),
                "fingerprint": self.framework_fingerprint,
            },
            "sections": sections,
            "overall_score": clamp_score(overall_score),
            "status": status,
            "control_summary": {
                "total": len(all_controls),
                "passed": sum(1 for control in all_controls if control.status == "pass"),
                "warnings": len(warning_controls),
                "failed": sum(1 for control in all_controls if control.status == "fail"),
                "errors": sum(1 for control in all_controls if control.status == "error"),
            },
            "critical_controls": [control.control_id for control in failed_controls if control.severity in {"critical", "high"}],
            "recommendations": self._build_recommendations(status, failed_controls, warning_controls),
            "metadata": sanitize_for_logging({
                "config_fingerprint": fingerprint(self.compliance_config),
                "weights": self.weights,
                "thresholds": self.report_thresholds,
            }),
        }
        self._store_evaluation(results)
        return sanitize_for_logging(results)

    def _evaluate_control_record(self, control: Mapping[str, Any], section_id: str = "", section_title: str = "") -> ControlEvaluation:
        start = time.perf_counter()
        started_at = utc_iso()
        control_id = normalize_text(control.get("controlId", "UNKNOWN"), max_length=96)
        evaluator = self.control_evaluators.get(control_id, self._evaluate_generic_control)
        try:
            result = evaluator(control, section_id, section_title)
            if not isinstance(result, ControlEvaluation):
                raise SystemIntegrityError(
                    component="compliance_checker",
                    anomaly_description="Control evaluator returned an invalid result type.",
                    expected_state="ControlEvaluation",
                    actual_state=type(result).__name__,
                )
            return result
        except SecurityError as exc:
            logger.warning("Structured compliance control failure: %s", stable_json(safe_log_payload(
                "compliance_control.security_error",
                {"control_id": control_id, "error": exc.to_log_record() if hasattr(exc, "to_log_record") else str(exc)},
            )))
            return self._control_result(
                control,
                status="error",
                score=0.0,
                severity="high",
                section_id=section_id,
                section_title=section_title,
                evaluator=evaluator.__name__,
                findings=["Structured security error occurred during control evaluation."],
                evidence=[EvidenceItem("security_error", "Control evaluation raised a structured security error.", "error", metadata={"error_type": type(exc).__name__})],
                remediation=["Review the structured incident record and repair the failing control evidence path."],
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:
            wrapped = wrap_security_exception(
                exc,
                operation="evaluate_compliance_control",
                component="compliance_checker",
                context={"control_id": control_id, "control": sanitize_for_logging(control)},
                error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                severity=SecuritySeverity.HIGH,
            )
            logger.warning("Compliance control evaluation failed: %s", stable_json(safe_log_payload(
                "compliance_control.unhandled_error",
                {"control_id": control_id, "error": wrapped.to_log_record() if hasattr(wrapped, "to_log_record") else str(wrapped)},
            )))
            return self._control_result(
                control,
                status="error",
                score=0.0,
                severity="high",
                section_id=section_id,
                section_title=section_title,
                evaluator=evaluator.__name__,
                findings=["Unhandled error occurred during control evaluation."],
                evidence=[EvidenceItem("exception", "Control evaluation failed and was wrapped as a security incident.", "error", metadata={"error_type": type(exc).__name__})],
                remediation=["Fix the evaluator or malformed evidence and add a regression test."],
                started_at=started_at,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _evaluate_control(self, control: Dict) -> str:
        """Backwards-compatible status-only evaluation of one control."""

        printer.status("CHECK", "Evaluating control", "info")
        return self._evaluate_control_record(control).status

    # ------------------------------------------------------------------
    # Control evaluators
    # ------------------------------------------------------------------
    def _evaluate_generic_control(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        printer.status("CHECK", "Control check", "info")
        findings: List[str] = []
        evidence: List[EvidenceItem] = []
        required_tags = self._get_control_list(control, "required_tags")
        required_fields = self._get_control_list(control, "required_fields")
        status = "pass"
        score = 1.0

        if not required_tags and not required_fields:
            status = "warning"
            score = 0.6
            findings.append("Control has no required_tags or required_fields; marked for review rather than fail.")
            evidence.append(EvidenceItem("control_definition", "No explicit generic evidence requirements were configured.", "warning", fingerprint=fingerprint(control)))
        else:
            for tag in required_tags:
                entries = self._recall_memory(tag, top_k=1)
                if not entries:
                    status = "fail"
                    score = 0.0
                    findings.append(f"Missing required evidence tag: {tag}.")
                    evidence.append(EvidenceItem("secure_memory", f"Missing required tag '{tag}'.", "missing", fingerprint=fingerprint(tag)))
                    continue
                data = self._entry_data(entries[0])
                evidence.append(EvidenceItem("secure_memory", f"Found required tag '{tag}'.", "observed", fingerprint=fingerprint(data)))
                if required_fields:
                    for field in required_fields:
                        value = get_nested(data, field, None) if isinstance(data, Mapping) else None
                        if value in (None, "", [], {}):
                            status = "fail"
                            score = 0.0
                            findings.append(f"Missing or empty required field '{field}' for tag '{tag}'.")
                            evidence.append(EvidenceItem("secure_memory", f"Required field '{field}' missing for tag '{tag}'.", "missing"))

        return self._control_result(
            control,
            status=status,
            score=score,
            severity=self._severity_for_status(status, control),
            section_id=section_id,
            section_title=section_title,
            evaluator="_evaluate_generic_control",
            findings=findings or ["Generic evidence requirements satisfied."],
            evidence=evidence,
            remediation=self._remediation_for_status(status, control),
        )

    def _generic_control_check(self, control: Dict) -> str:
        """Backwards-compatible generic control check returning status only."""

        return self._evaluate_generic_control(control, "", "").status

    def _evaluate_data_classification(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        classification_map = self._latest_memory_data("data_classification")
        required_fields = self._get_control_list(control, "required_fields") or list(self._cfg("control_requirements.data_classification.required_fields", []))
        allowed_labels = set(self._get_control_list(control, "allowed_labels") or list(self._cfg("control_requirements.data_classification.allowed_labels", [])))
        findings: List[str] = []
        evidence: List[EvidenceItem] = []
        status = "pass"

        if not isinstance(classification_map, Mapping):
            status = "fail"
            findings.append("Data classification map is missing from secure memory.")
        else:
            evidence.append(EvidenceItem("secure_memory:data_classification", "Data classification map found.", "observed", fingerprint=fingerprint(classification_map)))
            for field_name in required_fields:
                label = classification_map.get(field_name)
                if label not in allowed_labels:
                    status = "fail"
                    findings.append(f"Field '{field_name}' is missing or has an unapproved classification label.")

        return self._control_result(
            control,
            status=status,
            score=1.0 if status == "pass" else 0.0,
            severity=self._severity_for_status(status, control),
            section_id=section_id,
            section_title=section_title,
            evaluator="_evaluate_data_classification",
            findings=findings or ["Required data assets have approved classification labels."],
            evidence=evidence,
            remediation=self._remediation_for_status(status, control),
        )

    def _check_data_classification(self, control: Dict) -> str:
        return self._evaluate_data_classification(control, "", "").status

    def _evaluate_gdpr(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        checks = {
            "lawful_basis_consent": bool(get_nested(self._latest_memory_data("consent_records"), "consent_granted", False)),
            "purpose_limitation": bool(get_nested(self._latest_memory_data("data_usage_purpose"), "declared_purpose", "")),
            "data_minimization": self._evaluate_data_minimization(control, section_id, section_title).status == "pass",
            "subject_access": bool(get_nested(self._latest_memory_data("subject_requests"), "accessed", False)),
            "subject_correction": bool(get_nested(self._latest_memory_data("subject_requests"), "corrected", False)),
            "subject_deletion": bool(get_nested(self._latest_memory_data("subject_requests"), "deleted", False)),
            "retention_policy": bool(get_nested(self._latest_memory_data("retention_policy"), "expiration_days", None)),
        }
        passed = sum(1 for value in checks.values() if value)
        score = passed / len(checks)
        status = "pass" if score >= 1.0 else "warning" if score >= clamp_score(self._cfg("gdpr.partial_score_threshold", 0.75)) else "fail"
        findings = [f"{name}: {'satisfied' if value else 'missing'}" for name, value in checks.items()]
        evidence = [EvidenceItem("secure_memory", "GDPR evidence checks completed.", status, fingerprint=fingerprint(checks), metadata={"checks": checks})]
        return self._control_result(
            control,
            status=status,
            score=score,
            severity=self._severity_for_status(status, control),
            section_id=section_id,
            section_title=section_title,
            evaluator="_evaluate_gdpr",
            findings=findings,
            evidence=evidence,
            remediation=self._remediation_for_status(status, control),
        )

    def check_gdpr(self, control: Dict) -> str:
        return self._evaluate_gdpr(control, "", "").status

    def check_hipaa(self, data: Dict) -> str:
        """Backwards-compatible HIPAA check."""

        printer.status("CHECK", "Checking HIPAA compliance", "info")
        encrypted = coerce_bool(data.get("encrypted"), False) if isinstance(data, Mapping) else False
        contains_phi = bool(isinstance(data, Mapping) and ("PHI" in data or data.get("contains_phi")))
        return "pass" if not contains_phi or encrypted else "fail"

    def _evaluate_data_minimization(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        feature_log = self._latest_memory_data("feature_extraction")
        max_features = coerce_int(control.get("max_features", self._cfg("control_requirements.data_minimization.max_features", 30)), 30, minimum=1)
        input_limit = coerce_int(control.get("input_size_limit", get_nested(self.config, "adaptive_security.input_size_limit", self._cfg("control_requirements.data_minimization.input_size_limit", 2024))), 2024, minimum=1)
        findings: List[str] = []
        evidence: List[EvidenceItem] = []
        score = 1.0

        if not isinstance(feature_log, Mapping):
            findings.append("Feature extraction evidence is missing from secure memory.")
            return self._control_result(control, "fail", 0.0, self._severity_for_status("fail", control), section_id, section_title, "_evaluate_data_minimization", findings, evidence, self._remediation_for_status("fail", control))

        features_used = list(feature_log.get("features", []) or [])
        input_size = coerce_int(feature_log.get("input_size", 0), 0, minimum=0)
        evidence.append(EvidenceItem("secure_memory:feature_extraction", "Feature extraction evidence found.", "observed", fingerprint=fingerprint(feature_log), metadata={"feature_count": len(features_used), "input_size": input_size}))
        if not features_used:
            findings.append("No declared features found in feature extraction evidence.")
            score -= 0.5
        if len(features_used) > max_features:
            findings.append(f"Feature count {len(features_used)} exceeds configured maximum {max_features}.")
            score -= 0.4
        if input_size > input_limit:
            findings.append(f"Input size {input_size} exceeds configured limit {input_limit}.")
            score -= 0.4
        score = clamp_score(score)
        status = "pass" if score >= 0.95 else "warning" if score >= 0.6 else "fail"
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_data_minimization", findings or ["Feature usage and input size are within minimization limits."], evidence, self._remediation_for_status(status, control))

    def _check_data_minimization(self, control: Dict) -> str:
        return self._evaluate_data_minimization(control, "", "").status

    def _evaluate_model_integrity(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        trusted_hashes = self._load_trusted_hashes(control)
        model_paths = self._model_paths(control, trusted_hashes)
        findings: List[str] = []
        evidence: List[EvidenceItem] = []
        checked = 0
        passed = 0

        if not trusted_hashes:
            return self._control_result(control, "fail", 0.0, self._severity_for_status("fail", control), section_id, section_title, "_evaluate_model_integrity", ["Trusted model hashes are not configured or stored."], evidence, self._remediation_for_status("fail", control))

        for model_name, path in model_paths.items():
            expected_hash = trusted_hashes.get(model_name) or trusted_hashes.get(Path(model_name).name) or trusted_hashes.get(str(path))
            checked += 1
            if not path.exists() or not path.is_file():
                findings.append(f"Model file missing: {model_name}.")
                evidence.append(EvidenceItem("model_file", f"Model path missing for {model_name}.", "missing", fingerprint=fingerprint(str(path))))
                continue
            max_bytes = coerce_int(self._cfg("model_integrity.max_model_bytes", 20_000_000), 20_000_000, minimum=1024)
            if path.stat().st_size > max_bytes:
                findings.append(f"Model file {model_name} exceeds configured maximum size.")
                evidence.append(EvidenceItem("model_file", f"Model file too large for {model_name}.", "fail", fingerprint=fingerprint(str(path))))
                continue
            actual_hash = hash_bytes(path.read_bytes(), algorithm=str(self._cfg("model_integrity.hash_algorithm", "sha256")))
            evidence.append(EvidenceItem("model_hash", f"Model hash calculated for {model_name}.", "observed", fingerprint=actual_hash[:16], metadata={"model": model_name, "path_fingerprint": fingerprint(str(path))}))
            if not expected_hash:
                findings.append(f"No expected hash configured for model file: {model_name}.")
                continue
            if constant_time_equals(str(actual_hash), str(expected_hash)):
                passed += 1
            else:
                findings.append(f"Hash mismatch for model file: {model_name}.")

        score = passed / checked if checked else 0.0
        status = "pass" if checked and passed == checked else "fail"
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_model_integrity", findings or ["All configured model artifacts match trusted hashes."], evidence, self._remediation_for_status(status, control))

    def _check_model_integrity(self, control: Dict) -> str:
        return self._evaluate_model_integrity(control, "", "").status

    def _evaluate_application_security(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        required_sections = self._get_control_list(control, "required_config_sections") or list(self._cfg("control_requirements.application_security.required_config_sections", []))
        missing = [section for section in required_sections if not isinstance(self.config.get(section), Mapping) or not self.config.get(section)]
        checks = {
            "adaptive_security_configured": bool(get_nested(self.config, "adaptive_security.phishing_threshold", None)),
            "safety_guard_configured": bool(self.config.get("safety_guard")),
            "security_error_configured": bool(self.config.get("security_error")),
            "helper_configured": bool(self.config.get("safety_helpers", True)),
            "required_sections_present": not missing,
        }
        score = sum(1 for value in checks.values() if value) / len(checks)
        status = "pass" if score >= 1.0 else "warning" if score >= 0.7 else "fail"
        findings = [f"{name}: {'satisfied' if value else 'missing'}" for name, value in checks.items()]
        if missing:
            findings.append(f"Missing required config sections: {', '.join(missing)}.")
        evidence = [EvidenceItem("secure_config.yaml", "Application security configuration checks completed.", status, fingerprint=fingerprint(checks), metadata={"checks": checks})]
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_application_security", findings, evidence, self._remediation_for_status(status, control))

    def _evaluate_operational_security(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        stats = self.memory.get_statistics() if hasattr(self.memory, "get_statistics") else {}
        bootstrap_tags = ["retention_policy", "trusted_hashes", "data_usage_purpose"]
        present_tags = {tag: bool(self._recall_memory(tag, top_k=1)) for tag in bootstrap_tags}
        checks = {
            "memory_available": bool(self.memory),
            "retention_policy_present": present_tags["retention_policy"],
            "trusted_hashes_present": present_tags["trusted_hashes"],
            "data_usage_purpose_present": present_tags["data_usage_purpose"],
            "statistics_available": isinstance(stats, Mapping),
        }
        score = sum(1 for value in checks.values() if value) / len(checks)
        status = "pass" if score >= 1.0 else "warning" if score >= 0.7 else "fail"
        evidence = [EvidenceItem("secure_memory", "Operational secure-memory checks completed.", status, fingerprint=fingerprint(checks), metadata={"checks": checks, "stats": sanitize_for_logging(stats)})]
        findings = [f"{name}: {'satisfied' if value else 'missing'}" for name, value in checks.items()]
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_operational_security", findings, evidence, self._remediation_for_status(status, control))

    def _evaluate_audit_logging(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        audit_entries = self.memory.audit_access(max_results=10) if hasattr(self.memory, "audit_access") else []
        has_audit = isinstance(audit_entries, list)
        score = 1.0 if has_audit else 0.0
        status = "pass" if has_audit else "fail"
        evidence = [EvidenceItem("secure_memory.audit", "Audit access interface checked.", status, fingerprint=fingerprint(audit_entries[:3] if has_audit else []), metadata={"sample_count": len(audit_entries) if has_audit else 0})]
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_audit_logging", ["Audit logging interface is available." if has_audit else "Audit logging interface is unavailable."], evidence, self._remediation_for_status(status, control))

    def _evaluate_privacy_redaction(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        sample = "email alice@example.com token=SECRET123456789012345 password=hunter2"
        redacted = sanitize_for_logging({"sample": sample})
        rendered = stable_json(redacted)
        unsafe_markers = ["alice@example.com", "SECRET123456789012345", "hunter2"]
        leaked = [marker for marker in unsafe_markers if marker in rendered]
        status = "fail" if leaked else "pass"
        score = 0.0 if leaked else 1.0
        evidence = [EvidenceItem("safety_helpers.sanitize_for_logging", "Redaction smoke test executed.", status, fingerprint=fingerprint(rendered), metadata={"leaked_count": len(leaked)})]
        findings = ["Redaction leaked sensitive sample markers."] if leaked else ["Redaction smoke test did not expose sensitive sample markers."]
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_privacy_redaction", findings, evidence, self._remediation_for_status(status, control))

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_report(self, results: Dict) -> str:
        """Generate an audit-safe Markdown compliance report."""

        printer.status("CHECK", "Generating report", "info")
        safe_results = sanitize_for_logging(results)
        report = [
            "# Security Compliance Report",
            f"**Schema**: `{REPORT_SCHEMA_VERSION}`",
            f"**Generated**: {utc_iso()}",
            f"**Framework**: {get_nested(safe_results, 'framework.name', 'Safety Compliance Framework')}",
            f"**Framework Version**: {get_nested(safe_results, 'framework.version', 'unknown')}",
            f"**Framework Fingerprint**: `{get_nested(safe_results, 'framework.fingerprint', 'unknown')}`",
            f"**Overall Compliance Status**: {str(safe_results.get('status', 'unknown')).upper()}",
            f"**Overall Score**: {clamp_score(safe_results.get('overall_score', 0.0)):.1%}",
            "---",
        ]
        summary = safe_results.get("control_summary", {})
        if isinstance(summary, Mapping):
            report.extend([
                "## Control Summary",
                f"- Total: {summary.get('total', 0)}",
                f"- Passed: {summary.get('passed', 0)}",
                f"- Warnings: {summary.get('warnings', 0)}",
                f"- Failed: {summary.get('failed', 0)}",
                f"- Errors: {summary.get('errors', 0)}",
            ])

        for section_id, section_data in (safe_results.get("sections", {}) or {}).items():
            report.append(f"## {section_data.get('title', section_id)}")
            report.append(f"**Status**: {str(section_data.get('status', 'unknown')).upper()}")
            report.append(f"**Score**: {clamp_score(section_data.get('score', 0.0)):.1%}")
            report.append("### Control Status")
            for control in section_data.get("results", []):
                status = str(control.get("status", "unknown"))
                icon = "✅" if status == "pass" else "⚠️" if status == "warning" else "❌"
                report.append(f"- {icon} **{control.get('control_id', control.get('controlId', 'UNKNOWN'))}**: {control.get('objective', 'No objective')} — `{status}` / score `{clamp_score(control.get('score', 0.0)):.2f}` / owner `{control.get('owner', 'unknown')}`")
                findings = control.get("findings", []) or []
                for finding in findings[: coerce_int(self._cfg("reporting.max_findings_per_control", 3), 3, minimum=1)]:
                    report.append(f"  - Finding: {finding}")
            report.append("---")

        report.append("## Recommendations")
        recommendations = safe_results.get("recommendations", []) or []
        if recommendations:
            report.extend(f"- {item}" for item in recommendations)
        else:
            report.append("- Maintain current security practices and schedule the next compliance review.")
        report.append(f"\n---\n*Report generated by {self.__class__.__name__} v{MODULE_VERSION}*")
        return "\n".join(report)

    # ------------------------------------------------------------------
    # Helper methods local to compliance orchestration
    # ------------------------------------------------------------------
    def _control_result(
        self,
        control: Mapping[str, Any],
        status: str,
        score: float,
        severity: str,
        section_id: str,
        section_title: str,
        evaluator: str,
        findings: Optional[List[str]] = None,
        evidence: Optional[List[EvidenceItem]] = None,
        remediation: Optional[List[str]] = None,
        *,
        started_at: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> ControlEvaluation:
        normalized_status = status if status in {"pass", "warning", "fail", "error"} else "error"
        return ControlEvaluation(
            control_id=normalize_text(control.get("controlId", "UNKNOWN"), max_length=96),
            objective=normalize_text(control.get("objective", "No objective provided"), max_length=512),
            status=normalized_status,
            score=clamp_score(score),
            severity=normalize_identifier(severity or self._severity_for_status(normalized_status, control), max_length=32),
            owner=normalize_text(control.get("owner", self._cfg("default_owner", "security_compliance")), max_length=128),
            section_id=normalize_text(section_id or control.get("sectionId", "unknown"), max_length=96),
            section_title=normalize_text(section_title or control.get("sectionTitle", "Unsectioned Controls"), max_length=256),
            evaluator=evaluator,
            findings=[normalize_text(item, max_length=512) for item in (findings or [])],
            evidence=list(evidence or []),
            remediation=[normalize_text(item, max_length=512) for item in (remediation or [])],
            tags=dedupe_preserve_order([normalize_identifier(tag, max_length=64) for tag in (control.get("tags", []) or [])]),
            started_at=started_at or utc_iso(),
            completed_at=utc_iso(),
            duration_ms=float(duration_ms),
        )

    def _iter_section_controls(self, section: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        controls: List[Mapping[str, Any]] = []
        for control in section.get("controls", []) or []:
            if isinstance(control, Mapping):
                controls.append(control)
        for subsection in section.get("subsections", []) or []:
            if not isinstance(subsection, Mapping):
                continue
            for control in subsection.get("controls", []) or []:
                if isinstance(control, Mapping):
                    enriched = dict(control)
                    enriched.setdefault("subsectionId", subsection.get("subsectionId"))
                    enriched.setdefault("subsectionTitle", subsection.get("title"))
                    controls.append(enriched)
        return controls

    def _get_control_list(self, control: Mapping[str, Any], key: str) -> List[str]:
        value = control.get(key)
        if value is None:
            value = get_nested(self.compliance_config, f"control_requirements.{normalize_identifier(str(control.get('controlId', 'unknown')).lower())}.{key}", [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    def _score_controls(self, controls: Sequence[ControlEvaluation]) -> float:
        if not controls:
            return 0.0
        control_scores = {control.control_id: control.score for control in controls}
        control_weights = {control.control_id: coerce_float(self._cfg(f"control_weights.{control.control_id}", 1.0), 1.0, minimum=0.0) for control in controls}
        return weighted_average(control_scores, control_weights, default=0.0)

    def _get_compliance_status(self, score: float) -> str:
        printer.status("CHECK", "Status compliance", "info")
        critical = clamp_score(self.report_thresholds.get("critical", 0.8))
        warning = clamp_score(self.report_thresholds.get("warning", 0.9))
        score_value = clamp_score(score)
        if score_value < critical:
            return "critical"
        if score_value < warning:
            return "warning"
        return "compliant"

    def _severity_for_status(self, status: str, control: Mapping[str, Any]) -> str:
        configured = control.get("severity")
        if configured:
            return normalize_identifier(configured, max_length=32)
        if status == "error":
            return "high"
        if status == "fail":
            return "high"
        if status == "warning":
            return "medium"
        return "low"

    def _remediation_for_status(self, status: str, control: Mapping[str, Any]) -> List[str]:
        configured = control.get("remediation") or control.get("recommendations")
        if configured:
            if isinstance(configured, str):
                return [configured]
            if isinstance(configured, Iterable):
                return [str(item) for item in configured]
        if status == "pass":
            return ["Maintain the current control evidence and continue periodic review."]
        if status == "warning":
            return ["Review the control evidence and close partial compliance gaps before the next release gate."]
        return ["Remediate the failed control, update evidence, and rerun compliance evaluation before production approval."]

    def _build_recommendations(self, status: str, failed: Sequence[ControlEvaluation], warnings: Sequence[ControlEvaluation]) -> List[str]:
        if status == "critical":
            recommendations = [
                "Block production promotion until critical compliance gaps are remediated.",
                "Open a security/compliance review and assign owners for failed high-severity controls.",
            ]
        elif status == "warning":
            recommendations = [
                "Address warning controls within the configured remediation window.",
                "Refresh evidence in SecureMemory and rerun the compliance check before release sign-off.",
            ]
        else:
            recommendations = [
                "Maintain current controls and schedule the next periodic compliance review.",
                "Continue monitoring model integrity, data minimization, privacy, and audit evidence.",
            ]
        for control in list(failed)[:5]:
            recommendations.append(f"Remediate failed control {control.control_id}: {control.objective}")
        for control in list(warnings)[:3]:
            recommendations.append(f"Review warning control {control.control_id}: {control.objective}")
        return dedupe_preserve_order(recommendations)

    def _store_evaluation(self, results: Mapping[str, Any]) -> None:
        self._store_memory(
            sanitize_for_logging(dict(results)),
            tags=list(self._cfg("memory.evaluation_tags", ["compliance_evaluation", "security", "governance"])),
            sensitivity=coerce_float(self._cfg("memory.evaluation_sensitivity", 0.7), 0.7, minimum=0.0, maximum=1.0),
            metadata={"result_fingerprint": fingerprint(results), "schema_version": EVALUATION_SCHEMA_VERSION},
        )

    def _store_memory(self, payload: Any, *, tags: List[str], sensitivity: float, metadata: Optional[Mapping[str, Any]] = None) -> None:
        try:
            try:
                self.memory.add(
                    payload,
                    tags=tags,
                    sensitivity=sensitivity,
                    purpose="compliance_evaluation",
                    owner="compliance_checker",
                    classification="confidential",
                    source="compliance_checker",
                    metadata=metadata,
                    ttl_seconds=coerce_int(self._cfg("memory.evaluation_ttl_seconds", 86400), 86400, minimum=0),
                )
            except TypeError:
                self.memory.add(payload, tags=tags, sensitivity=sensitivity)
        except Exception as exc:
            raise AuditLogFailureError(
                "secure_memory.compliance_checker",
                f"Failed to store compliance evidence: {type(exc).__name__}",
                component="compliance_checker",
                cause=exc,
            ) from exc

    def _recall_memory(self, tag: str, top_k: int = 1) -> List[Any]:
        try:
            try:
                return list(self.memory.recall(tag=tag, top_k=top_k, access_context=self._memory_context("recall")))
            except TypeError:
                return list(self.memory.recall(tag=tag, top_k=top_k))
        except SecurityError:
            raise
        except Exception as exc:
            raise wrap_security_exception(
                exc,
                operation="recall_compliance_evidence",
                component="compliance_checker",
                context={"tag": tag, "tag_fingerprint": fingerprint(tag)},
                error_type=SecurityErrorType.ACCESS_VIOLATION,
                severity=SecuritySeverity.MEDIUM,
            ) from exc

    def _memory_context(self, purpose: str) -> Dict[str, Any]:
        access_cfg = get_nested(self.config, "secure_memory.access_validation", {}) or {}
        return {
            "auth_token": "compliance_checker_internal_context",
            "access_level": coerce_int(access_cfg.get("internal_access_level", access_cfg.get("min_access_level", 2)), 2),
            "purpose": purpose,
            "principal": "compliance_checker",
            "component": "compliance_checker",
            "request_id": generate_request_id(),
        }

    def _entry_data(self, entry: Any) -> Any:
        if isinstance(entry, Mapping) and "data" in entry:
            return entry.get("data")
        return entry

    def _latest_memory_data(self, tag: str) -> Any:
        entries = self._recall_memory(tag, top_k=1)
        return self._entry_data(entries[0]) if entries else None

    def _load_trusted_hashes(self, control: Mapping[str, Any]) -> Dict[str, str]:
        configured = dict(get_nested(self.compliance_config, "model_integrity.trusted_hashes", {}) or {})
        control_hashes = dict(control.get("trusted_hashes", {}) or {})
        for memory_entry in self._recall_memory("trusted_hashes", top_k=50):
            memory_hashes = self._entry_data(memory_entry)
            if isinstance(memory_hashes, Mapping):
                configured.update({str(k): str(v) for k, v in memory_hashes.items()})
        configured.update({str(k): str(v) for k, v in control_hashes.items()})
        trusted_hashes_path = control.get("trusted_hashes_path") or self._cfg("model_integrity.trusted_hashes_path")
        path = self._resolve_path(trusted_hashes_path)
        if path and path.exists():
            loaded = parse_json_object(load_text_file(path, max_bytes=coerce_int(self._cfg("model_integrity.max_hash_file_bytes", 1_048_576), 1_048_576, minimum=1024)), context="trusted_hashes")
            configured.update({str(k): str(v) for k, v in loaded.items()})
        return configured

    def _model_paths(self, control: Mapping[str, Any], trusted_hashes: Mapping[str, str]) -> Dict[str, Path]:
        configured_paths = dict(get_nested(self.compliance_config, "model_integrity.model_paths", {}) or {})
        control_paths = dict(control.get("model_paths", {}) or {})
        configured_paths.update(control_paths)
        if not configured_paths and self.phishing_model_path:
            configured_paths[Path(str(self.phishing_model_path)).name] = self.phishing_model_path
        if not configured_paths:
            model_dir_raw = self._cfg("model_integrity.model_dir")
            model_dir = self._resolve_path(model_dir_raw) if model_dir_raw else None
            if model_dir:
                for model_name in trusted_hashes:
                    configured_paths[Path(model_name).name] = str(model_dir / Path(model_name).name)
        return {str(name): self._resolve_path(path) or Path(str(path)) for name, path in configured_paths.items()}


if __name__ == "__main__":
    print("\n=== Running Compliance Checker ===\n")
    printer.status("TEST", "Compliance Checker initialized", "info")

    checker = ComplianceChecker()

    # Add deterministic self-test evidence. This exercises the checker without
    # requiring external UI packages or production service dependencies.
    checker.memory.add({"email.body": "Confidential", "training_features_url": "Restricted", "log_entries": "Internal"}, tags=["data_classification"], sensitivity=0.8)
    checker.memory.add({"consent_granted": True, "timestamp": utc_iso()}, tags=["consent_records"], sensitivity=0.8)
    checker.memory.add({"declared_purpose": "Phishing detection and cybersecurity threat mitigation"}, tags=["data_usage_purpose"], sensitivity=0.8)
    checker.memory.add({"features": ["url_length", "has_ip", "domain_entropy"], "input_size": 512}, tags=["feature_extraction"], sensitivity=0.7)
    checker.memory.add({"accessed": True, "corrected": True, "deleted": True, "timestamp": utc_iso()}, tags=["subject_requests"], sensitivity=0.8)
    checker.memory.add({"expiration_days": 365, "auto_delete": True, "policy_start": utc_iso()}, tags=["retention_policy"], sensitivity=0.8)

    test_model_path = Path("/tmp/compliance_checker_test_model.json")
    test_model_path.write_text(stable_json({"model": "self_test", "version": 1}), encoding="utf-8")
    checker.phishing_model_path = str(test_model_path)
    checker.memory.add({test_model_path.name: hash_bytes(test_model_path.read_bytes())}, tags=["trusted_hashes"], sensitivity=0.9)
    checker.compliance_config.setdefault("model_integrity", {}).setdefault("model_paths", {})[test_model_path.name] = str(test_model_path)

    results = checker.evaluate_compliance()
    report = checker.generate_report(results)

    assert results["control_summary"]["total"] >= 4, "Expected configured controls to run"
    assert results["overall_score"] > 0.0, "Expected positive compliance score"
    assert "Security Compliance Report" in report, "Expected markdown report"
    assert "compliance_checker" in stable_json(results), "Expected metadata in results"
    assert "SECRET" not in stable_json(results), "Sensitive markers should not leak"
    assert checker.check_hipaa({"PHI": "present", "encrypted": True}) == "pass"
    assert checker.check_hipaa({"PHI": "present", "encrypted": False}) == "fail"

    print(report)
    print("\n=== Test ran successfully ===\n")
