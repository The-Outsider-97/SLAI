"""Runtime content/privacy guard for the SLAI Safety Agent subsystem."""
from __future__ import annotations

import json
import math
import re
import numpy as np

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from .secure_memory import SecureMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Safety Guard")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
ANALYSIS_SCHEMA_VERSION = "safety_guard.analysis.v4"
REPORT_SCHEMA_VERSION = "safety_guard.report.v3"


@dataclass(frozen=True)
class GuardPattern:
    name: str
    pattern: str
    category: str
    replacement: Optional[str] = None
    severity: str = "medium"
    weight: float = 1.0
    description: str = ""
    source: str = "config"
    flags: Tuple[str, ...] = ()
    compiled: re.Pattern[str] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self); data.pop("compiled", None); data["pattern_fingerprint"] = fingerprint(self.pattern); data.pop("pattern", None)
        return redact_value(data)


@dataclass(frozen=True)
class GuardFinding:
    finding_id: str
    category: str
    severity: str
    risk_score: float
    confidence: float
    action: str
    pattern_name: str
    pattern_fingerprint: str
    evidence_preview: str
    remediation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging({
            "finding_id": self.finding_id, "category": self.category, "severity": self.severity,
            "risk_score": clamp_score(self.risk_score), "confidence": clamp_score(self.confidence), "action": self.action,
            "pattern_name": self.pattern_name, "pattern_fingerprint": self.pattern_fingerprint,
            "evidence_preview": redact_text(self.evidence_preview, max_length=256), "remediation": redact_text(self.remediation, max_length=512),
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class SafetyAnalysis:
    schema_version: str
    module_version: str
    analysis_id: str
    text_fingerprint: str
    sanitized_text: str
    changed: bool
    depth: str
    decision: str
    risk_score: float
    risk_level: str
    findings: List[GuardFinding]
    protection_stack: List[str]
    privacy_config: Dict[str, Any]
    context_fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_iso)

    def to_dict(self, *, include_sanitized_text: bool = True) -> Dict[str, Any]:
        data = {
            "schema_version": self.schema_version, "module_version": self.module_version, "analysis_id": self.analysis_id,
            "text_fingerprint": self.text_fingerprint, "changed": self.changed, "depth": self.depth,
            "decision": self.decision, "risk_score": clamp_score(self.risk_score), "risk_level": self.risk_level,
            "findings": [value.to_dict() for value in self.findings], "protection_stack": redact_value(self.protection_stack),
            "privacy_config": redact_value(self.privacy_config), "context_fingerprint": self.context_fingerprint,
            "metadata": redact_value(self.metadata), "timestamp": self.timestamp,
        }
        if include_sanitized_text: data["sanitized_text"] = redact_text(self.sanitized_text, max_length=4096)
        return data


class SafetyGuard:
    """Primary user-content safety and privacy gate."""

    monitor_thread = True

    def __init__(self, privacy_params: Optional[Dict[str, Any]] = None, *, memory: Optional[SecureMemory] = None) -> None:
        self.config = load_global_config()
        self.complience_config = get_config_section("safety_guard")  # backwards compatibility
        self.guard_config = self.complience_config
        self.memory = memory or SecureMemory.shared()
        self.enabled = coerce_bool(self.guard_config.get("enabled", True), True)
        self.strict_config_validation = coerce_bool(self.guard_config.get("strict_config_validation", True), True)
        self.max_text_length = coerce_int(self.guard_config.get("max_text_length", 8192), 8192, minimum=1)
        self.max_context_length = coerce_int(self.guard_config.get("max_context_length", 16384), 16384, minimum=1)
        self.store_events = coerce_bool(self.guard_config.get("store_events", True), True)
        self.raise_on_block = coerce_bool(self.guard_config.get("raise_on_block", True), True)
        self.return_block_marker = str(self.guard_config.get("block_marker", "[SAFETY_BLOCK] Content violates safety policy"))
        self.allowed_depths = {str(v) for v in self.guard_config.get("allowed_depths", ["minimal", "balanced", "full"])}
        self.context_window = coerce_int(self.guard_config.get("context_window", 8), 8, minimum=1)
        self._session_history: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=self.context_window))
        self._state_lock = RLock()
        self.epsilon = coerce_float(self.guard_config.get("epsilon", 1.0), 1.0, minimum=1e-9)
        self.sensitivity = coerce_float(self.guard_config.get("sensitivity", 1.0), 1.0, minimum=0.0)
        self.mechanism = str(self.guard_config.get("mechanism", "laplace")).lower().strip()
        self.delta = coerce_float(self.guard_config.get("delta", 1e-5), 1e-5, minimum=1e-12, maximum=1.0)
        seed = self.guard_config.get("privacy_noise_seed")
        self._rng = np.random.default_rng(coerce_int(seed, 0) if seed is not None else None)
        if privacy_params:
            self.epsilon = coerce_float(privacy_params.get("epsilon", self.epsilon), self.epsilon, minimum=1e-9)
            self.sensitivity = coerce_float(privacy_params.get("sensitivity", self.sensitivity), self.sensitivity, minimum=0.0)
            self.mechanism = str(privacy_params.get("mechanism", self.mechanism)).lower().strip()
            self.delta = coerce_float(privacy_params.get("delta", self.delta), self.delta, minimum=1e-12, maximum=1.0)
        self._validate_configuration(); self._validate_privacy_params()
        self.redact_patterns = self._load_resource_patterns("pii_patterns_path", "pii_patterns", "pii")
        self.toxicity_patterns = self._load_resource_patterns("toxicity_patterns_path", "toxicity_patterns", "toxicity")
        self.authority_phrases = self._load_resource_patterns("authority_phrases_path", "authority_phrases", "authority_escalation")
        self.manipulation_patterns = self._load_resource_patterns("manipulation_patterns_path", "manipulation_patterns", "prompt_manipulation")
        self.group_targeting_patterns = self._load_resource_patterns("group_targeting_patterns_path", "group_targeting_patterns", "targeted_harassment")
        self.boundary_phrases = self._load_resource_patterns("boundary_phrases_path", "boundary_phrases", "boundary_testing")
        self.injection_patterns = self._normalize_inline_patterns(self.guard_config.get("prompt_injection_patterns") or self.guard_config.get("injection_patterns") or [], "prompt_injection")
        self.misinformation_patterns = self._normalize_inline_patterns(self.guard_config.get("misinformation_patterns") or [], "misinformation")
        self.conversation_risk_patterns = self._normalize_inline_patterns(self.guard_config.get("conversation_risk_patterns") or [], "conversation_risk")
        self.sensitive_contexts = self._load_sensitive_contexts()
        self._compiled_redaction = self._compile_guard_patterns(self.redact_patterns)
        self._compiled_toxicity = self._compile_guard_patterns(self.toxicity_patterns)
        self._compiled_authority = self._compile_guard_patterns(self.authority_phrases)
        self._compiled_manipulation = self._compile_guard_patterns(self.manipulation_patterns)
        self._compiled_group_targeting = self._compile_guard_patterns(self.group_targeting_patterns)
        self._compiled_boundary = self._compile_guard_patterns(self.boundary_phrases)
        self._compiled_injection = self._compile_guard_patterns(self.injection_patterns)
        self._compiled_misinformation = self._compile_guard_patterns(self.misinformation_patterns)
        self._compiled_conversation_risk = self._compile_guard_patterns(self.conversation_risk_patterns)

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.guard_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.guard_config, Mapping):
            raise ConfigurationTamperingError("safety_guard", "safety_guard config must be a mapping", component="safety_guard")
        thresholds = self.guard_config.get("thresholds", {}) or {}
        if clamp_score(thresholds.get("review", 0.45)) > clamp_score(thresholds.get("block", 0.74)):
            raise ConfigurationTamperingError("safety_guard.thresholds", "review threshold must be <= block threshold", component="safety_guard")

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _resolve_config_path(self, raw_path: Optional[str]) -> Optional[Path]:
        if not raw_path: return None
        candidate = Path(str(raw_path)).expanduser()
        if candidate.is_absolute(): return candidate
        possible = [candidate, Path.cwd()/candidate]
        config_path = self.config.get("__config_path__")
        if config_path:
            parent = Path(str(config_path)).resolve().parent
            possible.extend([parent/candidate, parent.parent/candidate, parent.parent.parent/candidate, parent.parent.parent.parent/candidate])
        possible.append(Path("/mnt/data")/candidate.name)
        for path in possible:
            if path.exists(): return path
        return possible[0]

    def _load_json_resource(self, path_key: str, inline_key: str, label: str) -> List[Any]:
        path = self._resolve_config_path(self.guard_config.get(path_key)); inline = self.guard_config.get(inline_key, [])
        if path and path.exists():
            try:
                loaded = json.loads(load_text_file(path, max_bytes=coerce_int(self.guard_config.get("max_pattern_file_bytes", 2_097_152), 2_097_152, minimum=1024)))
                if isinstance(loaded, list): return loaded
                if isinstance(loaded, Mapping) and isinstance(loaded.get("patterns"), list): return list(loaded["patterns"])
                raise ConfigurationTamperingError(str(path), f"{label} resource must be a list or object with patterns", component="safety_guard")
            except SecurityError: raise
            except Exception as exc:
                raise wrap_security_exception(exc, operation=f"load_{normalize_identifier(label)}", component="safety_guard", error_type=SecurityErrorType.CONFIGURATION_TAMPERING, severity=SecuritySeverity.HIGH) from exc
        if isinstance(inline, list) and inline: return inline
        if self.strict_config_validation:
            raise ConfigurationTamperingError(str(self.guard_config.get(path_key) or inline_key), f"No configured {label} resource found", component="safety_guard")
        return []

    def _load_resource_patterns(self, path_key: str, inline_key: str, category: str) -> List[GuardPattern]:
        return [self._normalize_pattern_item(item, category, index, inline_key) for index, item in enumerate(self._load_json_resource(path_key, inline_key, inline_key))]

    def _normalize_inline_patterns(self, values: Sequence[Any], category: str) -> List[GuardPattern]:
        return [self._normalize_pattern_item(item, category, index, "secure_config") for index, item in enumerate(values)]

    def _normalize_pattern_item(self, item: Any, default_category: str, index: int, source: str) -> GuardPattern:
        if isinstance(item, GuardPattern): return item
        if isinstance(item, str): return GuardPattern(f"{default_category}_{index}", self._strip_regex_literal(item), default_category, source=source)
        if not isinstance(item, Mapping):
            raise ConfigurationTamperingError("safety_guard.patterns", "Pattern entry must be string or mapping", component="safety_guard")
        raw = item.get("pattern") or item.get("phrase") or item.get("regex")
        if not raw: raise ConfigurationTamperingError("safety_guard.patterns", "Pattern entry missing pattern/phrase/regex", component="safety_guard")
        flags = item.get("flags", [])
        return GuardPattern(
            name=normalize_identifier(item.get("name") or item.get("id") or item.get("replacement") or f"{default_category}_{index}", max_length=96),
            pattern=self._strip_regex_literal(str(raw)), category=normalize_identifier(item.get("category") or default_category, max_length=96),
            replacement=str(item.get("replacement")) if item.get("replacement") is not None else None,
            severity=normalize_identifier(item.get("severity", "medium"), max_length=32), weight=coerce_float(item.get("weight", 1.0), 1.0, minimum=0.0, maximum=10.0),
            description=normalize_text(item.get("description", ""), max_length=512), source=source,
            flags=tuple(str(v) for v in flags) if isinstance(flags, Iterable) and not isinstance(flags, str) else (),
        )

    @staticmethod
    def _strip_regex_literal(pattern: str) -> str:
        value = pattern.strip()
        if len(value) >= 3 and value[0] in {"r", "R"} and value[1] in {"'", '"'} and value[-1] == value[1]: return value[2:-1]
        if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]: return value[1:-1]
        return value

    def _compile_guard_patterns(self, patterns: Sequence[GuardPattern]) -> List[GuardPattern]:
        output: List[GuardPattern] = []
        for value in patterns:
            flags = re.IGNORECASE | re.UNICODE
            for name in value.flags: flags |= getattr(re, name.upper(), 0)
            try: compiled = re.compile(value.pattern, flags)
            except re.error as exc:
                if self.strict_config_validation: raise ConfigurationTamperingError("safety_guard.patterns", f"Invalid regex: {exc}", component="safety_guard", context={"pattern": value.name}) from exc
                continue
            output.append(GuardPattern(**{**asdict(value), "compiled": compiled}))
        return output

    def _load_sensitive_contexts(self) -> Dict[str, List[GuardPattern]]:
        raw = self.guard_config.get("sensitive_contexts", {}) or {}
        if not isinstance(raw, Mapping): raise ConfigurationTamperingError("safety_guard.sensitive_contexts", "sensitive_contexts must be mapping", component="safety_guard")
        return {str(name): self._normalize_inline_patterns(entries if isinstance(entries, list) else [], f"context_{name}") for name, entries in raw.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, text: str, context: Optional[Mapping[str, Any]] = None, *, depth: str = "full") -> SafetyAnalysis:
        if not self.enabled: raise SecurityError(SecurityErrorType.POLICY_BYPASS_ATTEMPT, "SafetyGuard is disabled.", component="safety_guard", response_action=SecurityResponseAction.BLOCK)
        if not isinstance(text, str): raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "SafetyGuard input must be text.", component="safety_guard")
        normalized_depth = str(depth or "full").lower().strip()
        if normalized_depth not in self.allowed_depths: raise ConfigurationTamperingError("safety_guard.allowed_depths", f"Unsupported depth: {depth}", component="safety_guard")
        original = normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)
        context_map = dict(context or {}); context_text = self._context_text(context_map)
        findings: List[GuardFinding] = []; stack: List[str] = []
        sanitized = self._apply_redaction(original, stack, findings=findings)
        findings.extend(self._scan_patterns(original, self._compiled_injection, "prompt_injection"))
        findings.extend(self._scan_patterns(original, self._compiled_misinformation, "misinformation"))
        if normalized_depth in {"balanced", "full"}:
            findings.extend(self._scan_patterns(original, self._compiled_toxicity, "toxicity"))
            findings.extend(self._scan_patterns(original, self._compiled_manipulation, "prompt_manipulation"))
            findings.extend(self._scan_patterns(original, self._compiled_boundary, "boundary_testing"))
        if normalized_depth == "full":
            findings.extend(self._scan_patterns(original, self._compiled_authority, "authority_escalation"))
            findings.extend(self._scan_patterns(original, self._compiled_group_targeting, "targeted_harassment"))
            findings.extend(self._scan_patterns(context_text, self._compiled_conversation_risk, "conversation_risk"))
            sanitized = self._apply_sensitive_context_protection(sanitized, context_map, stack, findings)
        self._post_sanitization_checks(sanitized)
        findings = self._dedupe_findings(findings)
        risk = self._aggregate_findings(findings)
        decision = threshold_decision(risk, block_threshold=coerce_float(self._cfg("thresholds.block", 0.74), 0.74), review_threshold=coerce_float(self._cfg("thresholds.review", 0.45), 0.45))
        analysis = SafetyAnalysis(ANALYSIS_SCHEMA_VERSION, MODULE_VERSION, generate_identifier("guard"), fingerprint(original), sanitized, original != sanitized, normalized_depth, decision, risk, categorize_risk(risk), findings, dedupe_preserve_order(stack), self._public_privacy_config(), fingerprint(context_text) if context_text else None, {"input_length": len(original), "sanitized_length": len(sanitized), "finding_count": len(findings), "session_scoped_context": bool(self._session_key(context_map))})
        self._record_context("user", sanitized, context_map)
        self._store_analysis(analysis, context=context_map)
        self._raise_for_blocking_findings(analysis, original)
        return analysis

    def sanitize(self, text: str, depth: str = "full") -> str:
        return self.analyze(text, depth=depth).sanitized_text

    def get_protection_report(self, text: str) -> Dict[str, Any]:
        try: return redact_value(self.analyze(text, depth="full").to_dict(include_sanitized_text=True))
        except SecurityError as exc: return redact_value({"schema_version": REPORT_SCHEMA_VERSION, "status": "blocked", "error": exc.to_public_response(), "audit": exc.to_log_record(), "text_fingerprint": fingerprint(text) if isinstance(text, str) else None, "timestamp": utc_iso()})

    def is_compliant(self, text: str, standard: str = "gdpr") -> bool:
        """Compatibility content-level policy check, not legal certification."""
        standard_name = normalize_identifier(standard, max_length=64).lower(); pii = self._detect_pii(text); toxicity = self._scan_patterns(text, self._compiled_toxicity, "toxicity")
        if standard_name in {"gdpr", "privacy"}: return not pii
        if standard_name in {"hipaa", "phi"}: return not any(term in " ".join(pii).lower() for term in ("medical", "health", "patient", "ssn", "phone", "address"))
        if standard_name in {"ai_safety", "content_safety"}: return not toxicity
        return not pii and not toxicity

    def detect_prompt_injection(self, prompt: str) -> bool:
        findings = self._scan_patterns(normalize_text(prompt, max_length=self.max_text_length, preserve_newlines=True), self._compiled_injection, "prompt_injection")
        if findings:
            top = max(findings, key=lambda f: f.risk_score)
            raise PromptInjectionError(detected_pattern=top.pattern_name, original_prompt=prompt, injected_payload=top.evidence_preview, component="safety_guard", confidence=top.confidence, risk_score=top.risk_score)
        return False

    def detect_misinformation(self, content: str) -> bool:
        findings = self._scan_patterns(normalize_text(content, max_length=self.max_text_length, preserve_newlines=True), self._compiled_misinformation, "misinformation")
        if not findings: return False
        top = max(findings, key=lambda f: f.risk_score)
        if coerce_bool(self._cfg("misinformation.raise_on_match", False), False):
            raise MisinformationError(content=content, identified_falsehood=f"configured risk marker: {top.pattern_name}", confidence_of_falsehood=min(top.confidence, 0.6), source_of_correction=top.remediation or "verification_required", component="safety_guard", risk_score=top.risk_score)
        return True

    def apply_differential_privacy(self, value: float, dataset_size: int = 1) -> float:
        """Apply a configured DP mechanism; privacy accounting is owned elsewhere."""
        self._validate_privacy_params(); numeric = coerce_float(value, 0.0)
        sensitivity = self.sensitivity
        if coerce_bool(self._cfg("dp_sensitivity_is_per_record_mean", False), False):
            sensitivity /= max(coerce_int(dataset_size, 1, minimum=1), 1)
        clip_min = self.guard_config.get("dp_clip_min"); clip_max = self.guard_config.get("dp_clip_max")
        if clip_min is not None: numeric = max(coerce_float(clip_min), numeric)
        if clip_max is not None: numeric = min(coerce_float(clip_max), numeric)
        with self._state_lock:
            if self.mechanism == "laplace": noise = float(self._rng.laplace(0.0, sensitivity / self.epsilon))
            else: noise = float(self._rng.normal(0.0, sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon))
        result = numeric + noise
        if clip_min is not None: result = max(coerce_float(clip_min), result)
        if clip_max is not None: result = min(coerce_float(clip_max), result)
        return float(result)

    # ------------------------------------------------------------------
    # Detection internals
    # ------------------------------------------------------------------
    def _apply_redaction(self, text: str, stack: List[str], *, findings: List[GuardFinding]) -> str:
        sanitized = text
        for pattern in self._compiled_redaction:
            matches = list(pattern.compiled.finditer(sanitized))
            if not matches: continue
            findings.append(self._finding_from_pattern(pattern, matches[0], "pii", action="redact"))
            sanitized = pattern.compiled.sub(pattern.replacement or str(self.guard_config.get("default_redaction_marker", "[REDACTED]")), sanitized)
            stack.append(f"redacted:{pattern.name}")
        return sanitized

    def _scan_patterns(self, text: str, patterns: Sequence[GuardPattern], category: str) -> List[GuardFinding]:
        findings: List[GuardFinding] = []
        if not text: return findings
        for pattern in patterns:
            matches = list(pattern.compiled.finditer(text))
            if not matches: continue
            findings.append(self._finding_from_pattern(pattern, matches[0], category, metadata={"match_count": len(matches)}))
        return findings

    def _finding_from_pattern(self, pattern: GuardPattern, match: re.Match[str], category: str, *, action: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> GuardFinding:
        severity_weights = self._cfg("severity_weights", {}) or {}
        severity_score = clamp_score(severity_weights.get(pattern.severity, 0.5))
        risk = clamp_score(severity_score * min(max(pattern.weight, 0.0), 1.25))
        confidence = clamp_score(coerce_float(self._cfg(f"category_confidence.{category}", 0.82), 0.82))
        return GuardFinding(generate_identifier("finding"), category, pattern.severity, risk, confidence, action or self._action_for_category(category), pattern.name, fingerprint(pattern.pattern), redact_text(match.group(0), max_length=96), self._remediation_for_category(category), dict(metadata or {}))

    def _dedupe_findings(self, findings: Sequence[GuardFinding]) -> List[GuardFinding]:
        groups: Dict[Tuple[str, str], GuardFinding] = {}
        for finding in findings:
            # Pattern family + redacted evidence fingerprint prevent correlated duplicate inflation.
            key = (finding.category, fingerprint(finding.evidence_preview))
            current = groups.get(key)
            if current is None or finding.risk_score > current.risk_score: groups[key] = finding
        return list(groups.values())

    def _aggregate_findings(self, findings: Sequence[GuardFinding]) -> float:
        if not findings: return 0.0
        # Aggregate by category first to avoid treating correlated regex matches as independent.
        category_scores: Dict[str, float] = {}
        for finding in findings: category_scores[finding.category] = max(category_scores.get(finding.category, 0.0), finding.risk_score)
        return combine_risk_scores(*category_scores.values(), method=str(self.guard_config.get("risk_aggregation", "noisy_or")))

    def _apply_sensitive_context_protection(self, text: str, context: Mapping[str, Any], stack: List[str], findings: List[GuardFinding]) -> str:
        context_type = normalize_identifier(context.get("type") or context.get("context_type") or "", max_length=64)
        patterns = self.sensitive_contexts.get(context_type, [])
        if not patterns: return text
        compiled = self._compile_guard_patterns(patterns); sanitized = text
        for pattern in compiled:
            matches = list(pattern.compiled.finditer(sanitized))
            if matches:
                findings.append(self._finding_from_pattern(pattern, matches[0], f"context_{context_type}", action="redact"))
                sanitized = pattern.compiled.sub(pattern.replacement or "[REDACTED]", sanitized); stack.append(f"context_redacted:{pattern.name}")
        return sanitized

    def _detect_pii(self, text: str) -> List[str]:
        normalized = normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)
        return dedupe_preserve_order([pattern.name for pattern in self._compiled_redaction if pattern.compiled.search(normalized)])

    def _post_sanitization_checks(self, text: str) -> None:
        residual = self._detect_pii(text)
        if residual: raise PiiLeakageError(data_description=", ".join(residual[:10]), leakage_source="safety_guard.post_sanitization", suspected_impact="Residual configured PII pattern after redaction", component="safety_guard")

    def _action_for_category(self, category: str) -> str:
        return normalize_identifier((self._cfg("category_actions", {}) or {}).get(category, "review"), max_length=32)

    def _remediation_for_category(self, category: str) -> str:
        configured = get_nested(self.guard_config, f"category_remediation.{category}", None)
        if configured: return normalize_text(configured, max_length=512)
        defaults = {"pii": "Redact or minimize sensitive data before downstream processing.", "prompt_injection": "Reject or sandbox the instruction and preserve instruction hierarchy.", "toxicity": "Block unsafe content or route it to the appropriate user-safety response.", "misinformation": "Require external verification before treating the claim as factual."}
        return defaults.get(category, "Review the finding before downstream action.")

    def _raise_for_blocking_findings(self, analysis: SafetyAnalysis, original_text: str) -> None:
        if analysis.decision != "block" or not self.raise_on_block: return
        if not analysis.findings: raise ContentPolicyViolationError("safety_guard_block_threshold", original_text, details="Aggregate safety risk exceeded configured threshold.", component="safety_guard", risk_score=analysis.risk_score)
        top = max(analysis.findings, key=lambda f: f.risk_score); action = self._action_for_category(top.category)
        if action == "redact": return
        if top.category in {"toxicity", "targeted_harassment"}: raise ToxicContentError(pattern=top.pattern_name, content=original_text, classification_details={"risk_level": analysis.risk_level, "confidence": top.confidence}, component="safety_guard", risk_score=top.risk_score)
        if top.category in {"prompt_injection", "prompt_manipulation"}: raise PromptInjectionError(detected_pattern=top.pattern_name, original_prompt=original_text, injected_payload=top.evidence_preview, component="safety_guard", risk_score=top.risk_score, confidence=top.confidence)
        raise ContentPolicyViolationError(top.category, original_text, details=f"Safety guard blocked due to {top.pattern_name}.", component="safety_guard", risk_score=analysis.risk_score)

    # ------------------------------------------------------------------
    # Context, privacy and persistence
    # ------------------------------------------------------------------
    def _session_key(self, context: Mapping[str, Any]) -> Optional[str]:
        raw = context.get("session_id") or context.get("conversation_id")
        return fingerprint(raw, length=24) if raw else None

    def _record_context(self, role: str, text: str, context: Mapping[str, Any]) -> None:
        key = self._session_key(context)
        if not key: return
        with self._state_lock:
            self._session_history[key].append({"role": normalize_identifier(role), "content": redact_text(text, max_length=512)})
            max_sessions = coerce_int(self._cfg("max_context_sessions", 1000), 1000, minimum=1)
            while len(self._session_history) > max_sessions: self._session_history.pop(next(iter(self._session_history)))

    def _context_text(self, context: Mapping[str, Any]) -> str:
        parts: List[str] = []
        for key in ("conversation", "context", "history", "previous_messages", "dialogue"):
            if context.get(key): parts.append(stable_json(context[key]) if not isinstance(context[key], str) else str(context[key]))
        session = self._session_key(context)
        if session:
            with self._state_lock:
                if session in self._session_history: parts.append(stable_json(list(self._session_history[session])))
        return normalize_text("\n".join(parts), max_length=self.max_context_length, preserve_newlines=True)

    def _validate_privacy_params(self) -> None:
        if self.epsilon <= 0: raise ConfigurationTamperingError("safety_guard.epsilon", "epsilon must be positive", component="safety_guard")
        if self.sensitivity < 0: raise ConfigurationTamperingError("safety_guard.sensitivity", "sensitivity must be non-negative", component="safety_guard")
        if self.mechanism not in {"laplace", "gaussian"}: raise ConfigurationTamperingError("safety_guard.mechanism", f"Unsupported mechanism: {self.mechanism}", component="safety_guard")

    def _public_privacy_config(self) -> Dict[str, Any]:
        return {"epsilon": self.epsilon, "sensitivity": self.sensitivity, "mechanism": self.mechanism, "delta": self.delta if self.mechanism == "gaussian" else None, "scope": "mechanism_only_no_budget_accounting"}

    def _store_analysis(self, analysis: SafetyAnalysis, *, context: Mapping[str, Any]) -> None:
        if not self.store_events: return
        data = analysis.to_dict(include_sanitized_text=False); data["context_fingerprint"] = fingerprint(sanitize_for_logging(dict(context)))
        try:
            self.memory.add(data, tags=["safety_guard", "safety_analysis", analysis.decision, analysis.risk_level], sensitivity=0.75 if analysis.decision == "block" else 0.55, purpose="safety_guard_audit", owner="safety_guard", classification="restricted" if analysis.decision == "block" else "confidential", source="safety_guard", metadata={"analysis_id": analysis.analysis_id, "decision": analysis.decision, "eligible_for_compliance": True})
        except Exception as exc:
            raise AuditLogFailureError("safety_guard.secure_memory", f"Failed to store safety analysis: {type(exc).__name__}", component="safety_guard", cause=exc) from exc

    def is_minimal_viable(self) -> bool:
        try: self._validate_privacy_params()
        except SecurityError: return False
        return bool(self._compiled_redaction and self._compiled_toxicity and self._compiled_injection)

    def generate_report(self, text: str, context: Optional[Mapping[str, Any]] = None) -> str:
        try: payload = self.analyze(text, context=context, depth="full").to_dict(include_sanitized_text=True)
        except SecurityError as exc: payload = {"blocked_error": exc.to_audit_format(), "text_fingerprint": fingerprint(text)}
        return "\n".join(["# Safety Guard Report", f"**Generated**: {utc_iso()}", f"**Module Version**: {MODULE_VERSION}", "", "```json", stable_json(redact_value(payload)), "```"])


__all__ = ["MODULE_VERSION", "ANALYSIS_SCHEMA_VERSION", "REPORT_SCHEMA_VERSION", "GuardPattern", "GuardFinding", "SafetyAnalysis", "SafetyGuard"]
