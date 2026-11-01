"""Evidence-based compliance assurance for the SLAI Safety Agent subsystem."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from .secure_memory import SecureMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Security Compliance Checker")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
EVALUATION_SCHEMA_VERSION = "compliance_checker.evaluation.v4"
REPORT_SCHEMA_VERSION = "compliance_checker.report.v3"


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    summary: str
    status: str = "observed"
    fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging(asdict(self))


@dataclass(frozen=True)
class ControlEvaluation:
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
    mandatory: bool = False
    evidence_quality: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = clamp_score(data["score"])
        data["evidence"] = [item.to_dict() if isinstance(item, EvidenceItem) else sanitize_for_logging(item) for item in self.evidence]
        return sanitize_for_logging(data)


@dataclass(frozen=True)
class SectionEvaluation:
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
    """Evaluate configured internal controls from trusted SecureMemory evidence."""

    def __init__(self, *, memory: Optional[SecureMemory] = None) -> None:
        self.config = load_global_config()
        self.complience_config = get_config_section("compliance_checker")  # backwards-compatible attribute
        self.compliance_config = self.complience_config
        self.compliance_file_path = self.compliance_config.get("compliance_file_path")
        self.phishing_model_path = self.compliance_config.get("phishing_model_path")
        self.enable_memory_bootstrap = coerce_bool(self.compliance_config.get("enable_memory_bootstrap", True), True)
        self.report_thresholds = dict(self.compliance_config.get("report_thresholds", {}))
        self.weights = dict(self.compliance_config.get("weights", {}))
        self._validate_configuration()
        self.memory = memory or SecureMemory.shared()
        if self.enable_memory_bootstrap:
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

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.compliance_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.compliance_config, Mapping):
            raise ConfigurationTamperingError("compliance_checker", "compliance_checker config must be a mapping", component="compliance_checker")
        thresholds = self.compliance_config.get("report_thresholds", {}) or {}
        critical = clamp_score(thresholds.get("critical", 0.8))
        warning = clamp_score(thresholds.get("warning", 0.9))
        if critical <= 0.0 or warning <= 0.0 or critical > warning:
            raise ConfigurationTamperingError("compliance_checker.report_thresholds", "Expected 0 < critical <= warning <= 1", component="compliance_checker")
        if not isinstance(self.compliance_config.get("weights", {}), Mapping):
            raise ConfigurationTamperingError("compliance_checker.weights", "weights must be a mapping", component="compliance_checker")

    # ------------------------------------------------------------------
    # Framework loading
    # ------------------------------------------------------------------
    def _resolve_path(self, raw_path: Optional[Union[str, Path]]) -> Optional[Path]:
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        if path.is_absolute():
            return path
        candidates = [Path.cwd() / path]
        config_path = self.config.get("__config_path__")
        if config_path:
            parent = Path(str(config_path)).resolve().parent
            candidates.extend([parent / path, parent.parent / path, parent.parent.parent / path, parent.parent.parent.parent / path])
        candidates.append(Path("/mnt/data") / path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_compliance_framework(self) -> Dict[str, Any]:
        entries = self._recall_memory("compliance_framework", top_k=1, include_non_evidence=True)
        if entries:
            candidate = self._entry_data(entries[0])
            if isinstance(candidate, Mapping) and candidate.get("sections"):
                self._validate_framework(candidate)
                return dict(candidate)
        path = self._resolve_path(self.compliance_file_path)
        if path and path.exists():
            raw = load_text_file(path, max_bytes=coerce_int(self._cfg("max_framework_bytes", 1_048_576), 1_048_576, minimum=4096))
            try:
                framework = json.loads(raw)
            except Exception as exc:
                raise wrap_security_exception(exc, operation="load_compliance_framework", component="compliance_checker", error_type=SecurityErrorType.CONFIGURATION_TAMPERING, severity=SecuritySeverity.HIGH) from exc
            if not isinstance(framework, Mapping):
                raise ConfigurationTamperingError(str(path), "Compliance framework must be a JSON object", component="compliance_checker")
            self._validate_framework(framework)
            self.memory.add(
                dict(framework), tags=["compliance_framework", "control_definition"], sensitivity=0.45,
                purpose="compliance_framework_definition", owner="compliance_checker", source="framework_file",
                metadata={"synthetic": True, "eligible_for_compliance": False, "framework_fingerprint": fingerprint(framework)},
            )
            return dict(framework)
        inline = self.compliance_config.get("framework")
        if isinstance(inline, Mapping) and inline.get("sections"):
            self._validate_framework(inline)
            return dict(inline)
        raise ConfigurationTamperingError("compliance_checker.compliance_file_path", "No valid compliance framework found", component="compliance_checker")

    def _validate_framework(self, framework: Mapping[str, Any]) -> None:
        sections = framework.get("sections")
        if not isinstance(sections, list) or not sections:
            raise ConfigurationTamperingError("compliance_framework.sections", "Framework must contain sections", component="compliance_checker")
        seen: set[str] = set()
        for section in sections:
            if not isinstance(section, Mapping):
                raise ConfigurationTamperingError("compliance_framework.section", "Section must be a mapping", component="compliance_checker")
            for control in self._iter_section_controls(section):
                cid = normalize_identifier(control.get("controlId"), max_length=96)
                if not cid:
                    raise ConfigurationTamperingError("compliance_framework.controlId", "Control id is required", component="compliance_checker")
                if cid in seen:
                    raise ConfigurationTamperingError("compliance_framework.controlId", f"Duplicate control id: {cid}", component="compliance_checker")
                seen.add(cid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_compliance(self) -> Dict[str, Any]:
        sections_out: Dict[str, Any] = {}
        section_scores: Dict[str, float] = {}
        section_weights: Dict[str, float] = {}
        controls: List[ControlEvaluation] = []
        for section in self.compliance_framework.get("sections", []):
            sid = normalize_identifier(section.get("sectionId", "section"), max_length=96).upper()
            title = normalize_text(section.get("title", sid), max_length=256)
            evaluated = [self._evaluate_control_record(control, sid, title) for control in self._iter_section_controls(section)]
            score = self._score_controls(evaluated)
            status = self._get_compliance_status(score)
            weight = coerce_float(section.get("weight", self.weights.get(sid.lower(), self.weights.get(normalize_identifier(title), 1.0))), 1.0, minimum=0.0)
            sections_out[sid] = SectionEvaluation(sid, title, status, score, weight, evaluated).to_dict()
            section_scores[sid] = score
            section_weights[sid] = weight
            controls.extend(evaluated)

        overall = weighted_average(section_scores, section_weights, default=0.0) if section_scores else 0.0
        failed = [c for c in controls if c.status in {"fail", "error"}]
        warnings = [c for c in controls if c.status == "warning"]
        mandatory_failed = [c for c in failed if c.mandatory or c.severity in {"critical", "high"} and self._control_is_mandatory(c.control_id)]
        status = "critical" if mandatory_failed else self._get_compliance_status(overall)
        result = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "timestamp": utc_iso(),
            "framework": {
                "version": get_nested(self.compliance_framework, "documentInfo.version", "unknown"),
                "name": get_nested(self.compliance_framework, "documentInfo.name", get_nested(self.compliance_framework, "documentInfo.title", "Safety Compliance Framework")),
                "fingerprint": self.framework_fingerprint,
            },
            "assurance_scope": "Internal evidence-based control assessment; not legal certification.",
            "sections": sections_out,
            "overall_score": clamp_score(overall),
            "status": status,
            "mandatory_failures": [c.control_id for c in mandatory_failed],
            "control_summary": {
                "total": len(controls), "passed": sum(c.status == "pass" for c in controls),
                "warnings": len(warnings), "failed": sum(c.status == "fail" for c in controls), "errors": sum(c.status == "error" for c in controls),
            },
            "critical_controls": [c.control_id for c in failed if c.severity in {"critical", "high"}],
            "recommendations": self._build_recommendations(status, failed, warnings),
        }
        self._store_evaluation(result)
        return sanitize_for_logging(result)

    def _evaluate_control_record(self, control: Mapping[str, Any], section_id: str = "", section_title: str = "") -> ControlEvaluation:
        started = utc_iso(); timer = time.perf_counter()
        cid = normalize_identifier(control.get("controlId", "UNKNOWN"), max_length=96).upper()
        evaluator = self.control_evaluators.get(cid, self._evaluate_generic_control)
        try:
            result = evaluator(control, section_id, section_title)
            if not isinstance(result, ControlEvaluation):
                raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Compliance evaluator returned invalid type.", component="compliance_checker", context={"control_id": cid})
            return result
        except SecurityError as exc:
            return self._control_result(control, "error", 0.0, "high", section_id, section_title, evaluator.__name__, ["Structured security failure during control evaluation."], [EvidenceItem("security_error", "Control evaluation raised a structured incident.", "error", metadata={"error_type": type(exc).__name__})], ["Repair the evidence/evaluator path and rerun the control."], started_at=started, duration_ms=(time.perf_counter()-timer)*1000)
        except Exception as exc:
            logger.warning("Compliance control %s failed with %s", cid, type(exc).__name__)
            return self._control_result(control, "error", 0.0, "high", section_id, section_title, evaluator.__name__, ["Unhandled evaluator failure."], [EvidenceItem("exception", "Control evaluator failed.", "error", metadata={"error_type": type(exc).__name__})], ["Fix the evaluator and add a regression test."], started_at=started, duration_ms=(time.perf_counter()-timer)*1000)

    def _evaluate_control(self, control: Dict) -> str:
        return self._evaluate_control_record(control).status

    # ------------------------------------------------------------------
    # Evaluators
    # ------------------------------------------------------------------
    def _evaluate_generic_control(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        required_tags = self._get_control_list(control, "required_tags")
        required_fields = self._get_control_list(control, "required_fields")
        evidence: List[EvidenceItem] = []
        findings: List[str] = []
        if not required_tags and not required_fields:
            return self._control_result(control, "warning", 0.5, "medium", section_id, section_title, "_evaluate_generic_control", ["No explicit evidence requirements are configured."], [], ["Define evidence tags and required fields for this control."])
        passed = 0; total = max(len(required_tags), 1)
        for tag in required_tags:
            entries = self._recall_memory(tag, top_k=5)
            if not entries:
                findings.append(f"Missing eligible evidence tag: {tag}.")
                continue
            valid = False
            for entry in entries:
                data = self._entry_data(entry)
                if required_fields and isinstance(data, Mapping):
                    missing = [field for field in required_fields if get_nested(data, field, None) in (None, "", [], {})]
                    if missing:
                        continue
                valid = True
                evidence.append(EvidenceItem(f"secure_memory:{tag}", "Eligible evidence record found.", "observed", fingerprint=fingerprint(data), metadata=self._evidence_metadata(entry)))
                break
            if valid: passed += 1
            else: findings.append(f"Evidence for {tag} did not satisfy required fields/provenance.")
        score = passed / total
        status = "pass" if score >= 1.0 else "warning" if score >= 0.75 else "fail"
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_generic_control", findings or ["Configured evidence requirements are satisfied."], evidence, self._remediation_for_status(status, control))

    def _generic_control_check(self, control: Dict) -> str:
        return self._evaluate_generic_control(control, "", "").status

    def _evaluate_data_classification(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        data = self._latest_memory_data("data_classification")
        required = self._get_control_list(control, "required_fields") or list(self._cfg("control_requirements.data_classification.required_fields", []))
        labels = {str(v) for v in (self._get_control_list(control, "allowed_labels") or list(self._cfg("control_requirements.data_classification.allowed_labels", [])))}
        findings: List[str] = []
        if not isinstance(data, Mapping):
            status, score = "fail", 0.0
            findings.append("Eligible data-classification evidence is missing.")
        else:
            invalid = [name for name in required if data.get(name) not in labels]
            status, score = ("pass", 1.0) if not invalid else ("fail", 0.0)
            findings.extend([f"Field '{name}' is missing or uses an unapproved classification label." for name in invalid])
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_data_classification", findings or ["Required assets use approved classification labels."], [], self._remediation_for_status(status, control))

    def _check_data_classification(self, control: Dict) -> str:
        return self._evaluate_data_classification(control, "", "").status

    def _evaluate_gdpr(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        lawful = self._latest_memory_data("lawful_basis")
        if lawful is None:
            # Backward-compatible evidence source: consent is one lawful basis, not the only one.
            consent = self._latest_memory_data("consent_records")
            if isinstance(consent, Mapping) and coerce_bool(consent.get("consent_granted"), False):
                lawful = {"basis": "consent", "documented": True}
        allowed_bases = {"consent", "contract", "legal_obligation", "vital_interests", "public_task", "legitimate_interests"}
        basis = normalize_identifier(get_nested(lawful or {}, "basis", ""), max_length=64)
        checks = {
            "lawful_basis": isinstance(lawful, Mapping) and basis in allowed_bases and coerce_bool((lawful or {}).get("documented", True), True),
            "purpose_limitation": bool(get_nested(self._latest_memory_data("data_usage_purpose") or {}, "declared_purpose", "")),
            "data_minimization": self._evaluate_data_minimization(control, section_id, section_title).status in {"pass", "warning"},
            "retention_policy": isinstance(self._latest_memory_data("retention_policy"), Mapping),
            "subject_rights_process": isinstance(self._latest_memory_data("subject_requests"), Mapping),
        }
        score = sum(bool(v) for v in checks.values()) / len(checks)
        status = "pass" if score >= 1.0 else "warning" if score >= clamp_score(self._cfg("gdpr.partial_score_threshold", 0.75)) else "fail"
        findings = [f"{name}: {'evidenced' if value else 'missing'}" for name, value in checks.items()]
        evidence = [EvidenceItem("secure_memory", "GDPR-oriented internal control evidence evaluated.", status, fingerprint=fingerprint(checks), metadata={"lawful_basis": basis or "missing"})]
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_gdpr", findings, evidence, self._remediation_for_status(status, control))

    def check_gdpr(self, control: Dict) -> str:
        return self._evaluate_gdpr(control, "", "").status

    def check_hipaa(self, data: Dict) -> str:
        """Compatibility helper for a narrow PHI technical-safeguard check.

        This does not represent legal HIPAA certification. It requires evidence of
        encryption, access control, audit controls, integrity, authentication and
        transmission protection whenever PHI is present.
        """
        if not isinstance(data, Mapping): return "fail"
        contains_phi = coerce_bool(data.get("contains_phi"), "PHI" in data)
        if not contains_phi: return "pass"
        required = ["encrypted", "access_control", "audit_controls", "integrity_controls", "authentication", "transmission_security"]
        return "pass" if all(coerce_bool(data.get(key), False) for key in required) else "fail"

    def _evaluate_data_minimization(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        feature_log = self._latest_memory_data("feature_extraction")
        if not isinstance(feature_log, Mapping):
            return self._control_result(control, "fail", 0.0, self._severity_for_status("fail", control), section_id, section_title, "_evaluate_data_minimization", ["Eligible feature-extraction evidence is missing."], [], self._remediation_for_status("fail", control))
        max_features = coerce_int(control.get("max_features", self._cfg("control_requirements.data_minimization.max_features", 30)), 30, minimum=1)
        input_limit = coerce_int(control.get("input_size_limit", self._cfg("control_requirements.data_minimization.input_size_limit", 2024)), 2024, minimum=1)
        features = list(feature_log.get("features", []) or [])
        input_size = coerce_int(feature_log.get("input_size", 0), 0, minimum=0)
        score = 1.0; findings: List[str] = []
        if not features: score -= 0.5; findings.append("No declared features were recorded.")
        if len(features) > max_features: score -= 0.4; findings.append("Feature count exceeds configured minimization limit.")
        if input_size > input_limit: score -= 0.4; findings.append("Input size exceeds configured minimization limit.")
        score = clamp_score(score)
        status = "pass" if score >= 0.95 else "warning" if score >= 0.6 else "fail"
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_data_minimization", findings or ["Feature use and input size are within configured minimization bounds."], [EvidenceItem("secure_memory:feature_extraction", "Feature-extraction evidence observed.", status, fingerprint(feature_log), {"feature_count": len(features), "input_size": input_size})], self._remediation_for_status(status, control))

    def _check_data_minimization(self, control: Dict) -> str:
        return self._evaluate_data_minimization(control, "", "").status

    def _evaluate_model_integrity(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        trusted = self._load_trusted_hashes(control)
        paths = self._model_paths(control, trusted)
        if not trusted:
            return self._control_result(control, "fail", 0.0, "high", section_id, section_title, "_evaluate_model_integrity", ["Trusted model hashes are not configured in eligible evidence/config."], [], self._remediation_for_status("fail", control))
        checked = 0; passed = 0; findings: List[str] = []; evidence: List[EvidenceItem] = []
        for model_name, path in paths.items():
            checked += 1
            expected = trusted.get(model_name) or trusted.get(Path(model_name).name) or trusted.get(str(path))
            if not expected or not path.is_file():
                findings.append(f"Model integrity evidence unavailable for {model_name}.")
                continue
            actual = self._hash_file(path)
            if str(actual).lower() == str(expected).lower():
                passed += 1; evidence.append(EvidenceItem("model_file", f"Integrity hash matched for {model_name}.", "observed", fingerprint=str(actual)))
            else:
                findings.append(f"Integrity hash mismatch for {model_name}.")
        score = passed / checked if checked else 0.0
        status = "pass" if checked and passed == checked else "fail"
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_model_integrity", findings or ["Configured model integrity hashes matched."], evidence, self._remediation_for_status(status, control))

    def _check_model_integrity(self, control: Dict) -> str:
        return self._evaluate_model_integrity(control, "", "").status

    def _hash_file(self, path: Path) -> str:
        import hashlib
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""): digest.update(chunk)
        return digest.hexdigest()

    def _evaluate_application_security(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        data = self._latest_memory_data("application_security") or self._latest_memory_data("security_scan")
        if not isinstance(data, Mapping):
            return self._control_result(control, "warning", 0.5, "medium", section_id, section_title, "_evaluate_application_security", ["No eligible application-security scan evidence found."], [], self._remediation_for_status("warning", control))
        critical = coerce_int(data.get("critical", data.get("critical_findings", 0)), 0, minimum=0)
        high = coerce_int(data.get("high", data.get("high_findings", 0)), 0, minimum=0)
        status = "pass" if critical == 0 and high == 0 else "fail"
        score = 1.0 if status == "pass" else 0.0
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_application_security", [f"Critical findings: {critical}; high findings: {high}."], [EvidenceItem("security_scan", "Application-security scan evidence evaluated.", status, fingerprint(data))], self._remediation_for_status(status, control))

    def _evaluate_operational_security(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        evidence_data = self._latest_memory_data("operational_security")
        required = self._get_control_list(control, "required_fields") or list(self._cfg("control_requirements.operational_security.required_fields", []))
        if not isinstance(evidence_data, Mapping):
            return self._control_result(control, "fail", 0.0, "high", section_id, section_title, "_evaluate_operational_security", ["Operational-security evidence is missing."], [], self._remediation_for_status("fail", control))
        missing = [field for field in required if not coerce_bool(get_nested(evidence_data, field, False), False)]
        status = "pass" if not missing else "fail"
        return self._control_result(control, status, 1.0 if status == "pass" else 0.0, self._severity_for_status(status, control), section_id, section_title, "_evaluate_operational_security", [f"Missing operational control: {field}" for field in missing] or ["Configured operational controls are evidenced."], [EvidenceItem("secure_memory:operational_security", "Operational evidence evaluated.", status, fingerprint(evidence_data))], self._remediation_for_status(status, control))

    def _evaluate_audit_logging(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        audit_entries = self.memory.audit_access(max_results=coerce_int(self._cfg("audit_logging.sample_size", 25), 25, minimum=1))
        required_fields = {"event_id", "timestamp", "event_type", "action", "allowed", "reason", "principal", "purpose"}
        structurally_valid = [entry for entry in audit_entries if isinstance(entry, Mapping) and required_fields.issubset(entry.keys())]
        minimum = coerce_int(self._cfg("audit_logging.min_events", 1), 1, minimum=1)
        status = "pass" if len(structurally_valid) >= minimum else "fail"
        score = min(len(structurally_valid) / minimum, 1.0)
        findings = [f"Observed {len(structurally_valid)} structurally valid audit events; minimum required is {minimum}."]
        return self._control_result(control, status, score, self._severity_for_status(status, control), section_id, section_title, "_evaluate_audit_logging", findings, [EvidenceItem("secure_memory.audit", "Audit trail structure and event presence checked.", status, fingerprint(audit_entries[:5]), {"event_count": len(audit_entries), "valid_count": len(structurally_valid)})], self._remediation_for_status(status, control))

    def _evaluate_privacy_redaction(self, control: Mapping[str, Any], section_id: str, section_title: str) -> ControlEvaluation:
        sample = "email alice@example.com token=SECRET123456789012345 password=hunter2"
        rendered = stable_json(sanitize_for_logging({"sample": sample}))
        leaked = [value for value in ["alice@example.com", "SECRET123456789012345", "hunter2"] if value in rendered]
        status = "fail" if leaked else "pass"
        return self._control_result(control, status, 0.0 if leaked else 1.0, self._severity_for_status(status, control), section_id, section_title, "_evaluate_privacy_redaction", ["Redaction smoke test leaked sensitive markers."] if leaked else ["Redaction smoke test did not expose sensitive markers."], [EvidenceItem("safety_helpers.sanitize_for_logging", "Redaction smoke test executed.", status, fingerprint(rendered), {"leaked_count": len(leaked)})], self._remediation_for_status(status, control))

    # ------------------------------------------------------------------
    # Reporting and helper methods
    # ------------------------------------------------------------------
    def generate_report(self, results: Dict) -> str:
        safe = sanitize_for_logging(results or {})
        lines = [
            "# Security Compliance Report",
            f"**Schema**: `{REPORT_SCHEMA_VERSION}`",
            f"**Generated**: {utc_iso()}",
            f"**Overall Status**: {str(safe.get('status', 'unknown')).upper()}",
            f"**Overall Score**: {clamp_score(safe.get('overall_score', 0.0)):.1%}",
            "**Scope**: Internal evidence-based control assessment; not legal certification.",
            "---",
        ]
        for section_id, section in (safe.get("sections", {}) or {}).items():
            lines.extend([f"## {section.get('title', section_id)}", f"**Status**: {str(section.get('status', 'unknown')).upper()}", f"**Score**: {clamp_score(section.get('score', 0.0)):.1%}"])
            for control in section.get("results", []) or []:
                lines.append(f"- **{control.get('control_id', 'UNKNOWN')}** — {control.get('status', 'unknown')} / {clamp_score(control.get('score', 0.0)):.2f}")
            lines.append("---")
        lines.append("## Recommendations")
        lines.extend([f"- {value}" for value in safe.get("recommendations", [])] or ["- Maintain current controls and evidence freshness."])
        return "\n".join(lines)

    def _control_result(self, control: Mapping[str, Any], status: str, score: float, severity: str, section_id: str, section_title: str, evaluator: str, findings: Optional[List[str]] = None, evidence: Optional[List[EvidenceItem]] = None, remediation: Optional[List[str]] = None, *, started_at: Optional[str] = None, duration_ms: float = 0.0) -> ControlEvaluation:
        normalized = status if status in {"pass", "warning", "fail", "error"} else "error"
        return ControlEvaluation(
            control_id=normalize_text(control.get("controlId", "UNKNOWN"), max_length=96),
            objective=normalize_text(control.get("objective", "No objective provided"), max_length=512),
            status=normalized, score=clamp_score(score), severity=normalize_identifier(severity or self._severity_for_status(normalized, control), max_length=32),
            owner=normalize_text(control.get("owner", self._cfg("default_owner", "security_compliance")), max_length=128),
            section_id=normalize_text(section_id or control.get("sectionId", "unknown"), max_length=96), section_title=normalize_text(section_title or control.get("sectionTitle", "Unsectioned Controls"), max_length=256), evaluator=evaluator,
            findings=[normalize_text(v, max_length=512) for v in (findings or [])], evidence=list(evidence or []), remediation=[normalize_text(v, max_length=512) for v in (remediation or [])], tags=dedupe_preserve_order([normalize_identifier(v, max_length=64) for v in (control.get("tags", []) or [])]),
            started_at=started_at or utc_iso(), completed_at=utc_iso(), duration_ms=float(duration_ms), mandatory=coerce_bool(control.get("mandatory", self._control_is_mandatory(str(control.get("controlId", "")))), False), evidence_quality=self._evidence_quality(evidence or []),
        )

    def _evidence_quality(self, evidence: Sequence[EvidenceItem]) -> str:
        if not evidence: return "missing"
        observed = sum(item.status == "observed" for item in evidence)
        if observed == len(evidence): return "direct"
        if observed: return "mixed"
        return "weak"

    def _iter_section_controls(self, section: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        controls = [value for value in section.get("controls", []) or [] if isinstance(value, Mapping)]
        for subsection in section.get("subsections", []) or []:
            if isinstance(subsection, Mapping):
                for value in subsection.get("controls", []) or []:
                    if isinstance(value, Mapping): controls.append(dict(value))
        return controls

    def _get_control_list(self, control: Mapping[str, Any], key: str) -> List[str]:
        value = control.get(key)
        if value is None:
            value = self._cfg(f"control_requirements.{normalize_identifier(str(control.get('controlId', 'unknown')).lower())}.{key}", [])
        if value is None: return []
        if isinstance(value, str): return [value]
        if isinstance(value, Iterable): return [str(v) for v in value if str(v).strip()]
        return [str(value)]

    def _score_controls(self, controls: Sequence[ControlEvaluation]) -> float:
        if not controls: return 0.0
        scores = {c.control_id: c.score for c in controls}
        weights = {c.control_id: coerce_float(self._cfg(f"control_weights.{c.control_id}", 1.0), 1.0, minimum=0.0) for c in controls}
        return weighted_average(scores, weights, default=0.0)

    def _get_compliance_status(self, score: float) -> str:
        critical = clamp_score(self.report_thresholds.get("critical", 0.8)); warning = clamp_score(self.report_thresholds.get("warning", 0.9)); value = clamp_score(score)
        if value < critical: return "critical"
        if value < warning: return "warning"
        return "compliant"

    def _severity_for_status(self, status: str, control: Mapping[str, Any]) -> str:
        if control.get("severity"): return normalize_identifier(control.get("severity"), max_length=32)
        return "high" if status in {"error", "fail"} else "medium" if status == "warning" else "low"

    def _remediation_for_status(self, status: str, control: Mapping[str, Any]) -> List[str]:
        configured = control.get("remediation") or control.get("recommendations")
        if isinstance(configured, str): return [configured]
        if isinstance(configured, Iterable): return [str(v) for v in configured]
        if status == "pass": return ["Maintain current evidence and continue periodic review."]
        if status == "warning": return ["Close partial evidence gaps and rerun the control before release sign-off."]
        return ["Remediate the failed control, refresh trusted evidence, and rerun evaluation before production approval."]

    def _build_recommendations(self, status: str, failed: Sequence[ControlEvaluation], warnings: Sequence[ControlEvaluation]) -> List[str]:
        values: List[str] = []
        if status == "critical": values.append("Block production promotion until mandatory/critical control failures are remediated.")
        elif status == "warning": values.append("Resolve warning controls before the next production gate.")
        else: values.append("Maintain controls and evidence freshness through periodic review.")
        values.extend(f"Remediate {c.control_id}: {c.objective}" for c in list(failed)[:8])
        values.extend(f"Review {c.control_id}: {c.objective}" for c in list(warnings)[:5])
        return dedupe_preserve_order(values)

    def _control_is_mandatory(self, control_id: str) -> bool:
        configured = {normalize_identifier(v, max_length=96).upper() for v in self._cfg("mandatory_controls", []) or []}
        return normalize_identifier(control_id, max_length=96).upper() in configured

    # ------------------------------------------------------------------
    # Secure-memory evidence access
    # ------------------------------------------------------------------
    def _store_evaluation(self, results: Mapping[str, Any]) -> None:
        self.memory.add(sanitize_for_logging(dict(results)), tags=list(self._cfg("memory.evaluation_tags", ["compliance_evaluation", "security", "governance"])), sensitivity=coerce_float(self._cfg("memory.evaluation_sensitivity", 0.7), 0.7), ttl_seconds=coerce_int(self._cfg("memory.evaluation_ttl_seconds", 86400), 86400, minimum=0), purpose="compliance_evaluation", owner="compliance_checker", classification="confidential", source="compliance_checker", metadata={"eligible_for_compliance": True, "result_fingerprint": fingerprint(results)})

    def _memory_context(self, purpose: str) -> Dict[str, Any]:
        return self.memory.internal_context(purpose, principal="compliance_checker")

    def _recall_memory(self, tag: str, top_k: int = 1, *, include_non_evidence: bool = False) -> List[Any]:
        entries = list(self.memory.recall(tag=tag, top_k=max(top_k * 4, top_k), access_context=self._memory_context("recall")))
        if include_non_evidence: return entries[:top_k]
        eligible = [entry for entry in entries if self.memory.is_evidence_eligible(entry)]
        return eligible[:top_k]

    def _entry_data(self, entry: Any) -> Any:
        return entry.get("data") if isinstance(entry, Mapping) and "data" in entry else entry

    def _evidence_metadata(self, entry: Any) -> Dict[str, Any]:
        if not isinstance(entry, Mapping): return {}
        meta = entry.get("meta", {}) if isinstance(entry.get("meta", {}), Mapping) else {}
        return sanitize_for_logging({"created_at": meta.get("created_at"), "source": meta.get("source"), "owner": meta.get("owner"), "classification": meta.get("classification"), "revision": meta.get("revision")})

    def _latest_memory_data(self, tag: str) -> Any:
        entries = self._recall_memory(tag, top_k=1)
        return self._entry_data(entries[0]) if entries else None

    def _load_trusted_hashes(self, control: Mapping[str, Any]) -> Dict[str, str]:
        configured = dict(self._cfg("model_integrity.trusted_hashes", {}) or {})
        configured.update({str(k): str(v) for k, v in dict(control.get("trusted_hashes", {}) or {}).items()})
        for entry in self._recall_memory("trusted_hashes", top_k=50):
            data = self._entry_data(entry)
            if isinstance(data, Mapping): configured.update({str(k): str(v) for k, v in data.items()})
        path = self._resolve_path(control.get("trusted_hashes_path") or self._cfg("model_integrity.trusted_hashes_path"))
        if path and path.exists():
            loaded = json.loads(load_text_file(path, max_bytes=coerce_int(self._cfg("model_integrity.max_hash_file_bytes", 1_048_576), 1_048_576)))
            if isinstance(loaded, Mapping): configured.update({str(k): str(v) for k, v in loaded.items()})
        return configured

    def _model_paths(self, control: Mapping[str, Any], trusted_hashes: Mapping[str, str]) -> Dict[str, Path]:
        configured = dict(self._cfg("model_integrity.model_paths", {}) or {})
        configured.update(dict(control.get("model_paths", {}) or {}))
        if self.phishing_model_path:
            configured.setdefault(Path(str(self.phishing_model_path)).name, self.phishing_model_path)
        result: Dict[str, Path] = {}
        for name in trusted_hashes:
            raw = configured.get(name) or configured.get(Path(name).name)
            if raw:
                resolved = self._resolve_path(raw)
                if resolved: result[name] = resolved
        return result


__all__ = ["MODULE_VERSION", "EVALUATION_SCHEMA_VERSION", "REPORT_SCHEMA_VERSION", "EvidenceItem", "ControlEvaluation", "SectionEvaluation", "ComplianceChecker"]

if __name__ == "__main__":
    print("\n=== Running Compliance Checker ===\n")
    printer.status("TEST", "Compliance Checker initialized", "info")
    printer.section_header("Smoke test #1: Initialization")

    Checker = ComplianceChecker()

    printer.status("START", " Compliance Checker ready", "success" if Checker is not None else "error")

    printer.section_header("Smoke test #2: Public API")
    Evaluate = Checker.evaluate_compliance()

    printer.status("Public API", Evaluate, "success" if Evaluate.get("decision") in {"allow", "review"} else "error")
    print("\n== Task run successfully ==\n")