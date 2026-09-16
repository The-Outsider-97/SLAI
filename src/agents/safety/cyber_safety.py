"""Cyber-security analysis and event anomaly monitoring for Safety Agent."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from .modules.neural_network import NeuralNetwork
from .secure_memory import SecureMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Cyber Safety Module")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
ANALYSIS_SCHEMA_VERSION = "cyber_safety.analysis.v4"
EVENT_SCHEMA_VERSION = "cyber_safety.event.v3"
THREAT_SCHEMA_VERSION = "cyber_safety.threat.v3"


@dataclass(frozen=True)
class CyberFinding:
    finding_id: str
    category: str
    finding_type: str
    name: str
    description: str
    severity: float
    confidence: float
    match_count: int = 1
    evidence_preview: str = ""
    pattern_fingerprint: str = ""
    remediation: str = ""
    source: str = "cyber_safety"
    tags: Tuple[str, ...] = field(default_factory=tuple)
    references: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = clamp_score(data["severity"])
        data["confidence"] = clamp_score(data["confidence"], default=0.75)
        data["evidence_preview"] = redact_text(data.get("evidence_preview", ""), max_length=240)
        return sanitize_for_logging(data)


@dataclass(frozen=True)
class CyberAnalysisResult:
    schema_version: str
    analysis_id: str
    timestamp: str
    context: str
    input_fingerprint: str
    risk_score: float
    risk_level: str
    decision: str
    findings: List[CyberFinding]
    recommendations: List[str]
    model_score: Optional[float] = None
    rule_score: float = 0.0
    vulnerability_score: float = 0.0
    context_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging({
            "schema_version": self.schema_version, "module_version": MODULE_VERSION,
            "analysis_id": self.analysis_id, "timestamp": self.timestamp, "context": self.context,
            "input_fingerprint": self.input_fingerprint, "risk_score": clamp_score(self.risk_score),
            "risk_level": self.risk_level, "decision": self.decision,
            "findings": [value.to_dict() for value in self.findings], "recommendations": list(self.recommendations),
            "model_score": self.model_score, "rule_score": clamp_score(self.rule_score),
            "vulnerability_score": clamp_score(self.vulnerability_score), "context_score": clamp_score(self.context_score),
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class EventAnalysisResult:
    schema_version: str
    event_id: str
    timestamp: str
    event_type: str
    entity: str
    anomaly_score: float
    normalized_anomaly_score: float
    is_anomaly: bool
    risk_level: str
    decision: str
    reason: str
    contributors: List[str]
    feature_scores: Dict[str, float]
    sequence: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging({
            "schema_version": self.schema_version, "module_version": MODULE_VERSION, "event_id": self.event_id,
            "timestamp": self.timestamp, "event_type": self.event_type, "entity": fingerprint(self.entity),
            "anomaly_score": self.anomaly_score, "normalized_anomaly_score": clamp_score(self.normalized_anomaly_score),
            "is_anomaly": self.is_anomaly, "risk_level": self.risk_level, "decision": self.decision,
            "reason": self.reason, "contributors": list(self.contributors), "feature_scores": dict(self.feature_scores),
            "sequence": list(self.sequence), "metadata": self.metadata,
        })


@dataclass
class RunningStatistic:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def variance(self) -> float:
        return self.m2 / max(self.count - 1, 1) if self.count > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def z_score(self, value: float) -> float:
        return abs(value - self.mean) / self.std_dev if self.std_dev > 1e-9 else 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)


@dataclass(frozen=True)
class ThreatAssessmentResult:
    schema_version: str
    assessment_id: str
    timestamp: str
    component: str
    action: str
    network_zone: str
    risk_score: float
    overall_risk: str
    threats: Dict[str, List[str]]
    mitigations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging(asdict(self))


class CyberSafetyModule:
    """Cyber/system threat analysis, distinct from user-content SafetyGuard policy."""

    _EXCLUDED_CROSS_DOMAIN_CATEGORIES = {"data_privacy", "content_output_safety", "prompt_security", "toxicity", "pii"}

    def __init__(self, *, memory: Optional[SecureMemory] = None) -> None:
        self.config = load_global_config()
        self.cyber_config = get_config_section("cyber_safety")
        self._validate_configuration()
        self.memory = memory or SecureMemory.shared()
        self.anomaly_threshold = coerce_float(self.cyber_config.get("anomaly_threshold", 3.0), 3.0, minimum=0.01)
        self.max_log_history = coerce_int(self.cyber_config.get("max_log_history", 1000), 1000, minimum=10)
        self.qnn_feature_dim = coerce_int(self.cyber_config.get("qnn_feature_dim", 8), 8, minimum=2, maximum=4096)
        self.adaptive_centroid_lr = coerce_float(self.cyber_config.get("adaptive_centroid_lr", 0.01), 0.01, minimum=0.0, maximum=1.0)
        self.use_centroid = coerce_bool(self.cyber_config.get("qnn_inspired_anomaly", False), False)
        self.security_rules = self._load_security_rules()
        self.vulnerability_signatures = self._load_vulnerability_signatures()
        self.compiled_rule_patterns = self._compile_rule_patterns(self.security_rules.get("patterns", []))
        self.compiled_vulnerability_patterns = self._compile_vulnerability_signatures(self.vulnerability_signatures)
        sequence_window = coerce_int(get_nested(self.cyber_config, "event_stream.sequence_window", 10), 10, minimum=3)
        self.event_log_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_log_history)
        self.sequence_patterns: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=sequence_window))
        self.sequence_counts: Dict[str, int] = defaultdict(int)
        self.event_statistics: Dict[str, RunningStatistic] = defaultdict(RunningStatistic)
        self.centroids: Dict[str, np.ndarray] = {}
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._state_lock = RLock()
        self.model: Optional[NeuralNetwork] = self._load_security_model()

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.cyber_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.cyber_config, Mapping):
            raise ConfigurationTamperingError("cyber_safety", "cyber_safety config must be a mapping", component="cyber_safety")
        thresholds = self.cyber_config.get("risk_thresholds", {}) or {}
        if clamp_score(thresholds.get("review", 0.5)) > clamp_score(thresholds.get("block", 0.75)):
            raise ConfigurationTamperingError("cyber_safety.risk_thresholds", "review must be <= block", component="cyber_safety")

    # ------------------------------------------------------------------
    # Resources and model
    # ------------------------------------------------------------------
    def _load_security_rules(self) -> Dict[str, Any]:
        inline = self.cyber_config.get("security_rules") if isinstance(self.cyber_config.get("security_rules"), Mapping) else {"principles": [], "patterns": []}
        assert inline is not None
        merged = {"principles": list(inline.get("principles", [])), "patterns": list(inline.get("patterns", []))}
        configured_path = self.cyber_config.get("cyber_rules_path")
        file_rules = self._load_json_resource(configured_path, "Cyber Security Rules", dict, required=False)
        if isinstance(file_rules, Mapping):
            merged["principles"] = dedupe_preserve_order(merged["principles"] + list(file_rules.get("principles", [])))
            merged["patterns"].extend(list(file_rules.get("patterns", [])))
        allow_cross_domain = coerce_bool(self._cfg("allow_cross_domain_patterns", False), False)
        for source in self.cyber_config.get("pattern_sources", []) or []:
            if not isinstance(source, Mapping): continue
            category = normalize_identifier(source.get("category", "cyber_threat"), max_length=64)
            if not allow_cross_domain and category in self._EXCLUDED_CROSS_DOMAIN_CATEGORIES:
                continue
            source_patterns = self._load_json_resource(source.get("path"), str(source.get("name", "Pattern Source")), list, required=coerce_bool(source.get("required", False), False))
            regexes = [str(item.get("pattern")) for item in source_patterns if isinstance(item, Mapping) and item.get("pattern")] + [str(item) for item in source_patterns if isinstance(item, str)]
            if regexes:
                merged["patterns"].append({"name": source.get("name", "External Pattern Source"), "category": category, "regex_list": regexes, "severity": source.get("severity", 0.6), "confidence": source.get("confidence", 0.75), "remediation": source.get("remediation", "Review the matched cyber pattern."), "tags": list(source.get("tags", []))})
        return merged

    def _load_vulnerability_signatures(self) -> Dict[str, Any]:
        loaded = self._load_json_resource(self.cyber_config.get("vulnerability_signatures_path"), "Vulnerability Signatures", dict, required=False)
        if loaded: return dict(loaded)
        inline = self.cyber_config.get("vulnerability_signatures", {})
        return dict(inline) if isinstance(inline, Mapping) else {}

    def _load_json_resource(self, file_path: Optional[Union[str, Path]], name: str, expected_type: type, *, required: bool) -> Any:
        if not file_path:
            if required: raise ConfigurationTamperingError("cyber_safety.resource", f"Missing required resource: {name}", component="cyber_safety")
            return {} if expected_type is dict else []
        path = Path(str(file_path)).expanduser()
        try:
            parsed = json.loads(load_text_file(path, max_bytes=coerce_int(self._cfg("resource_loading.max_json_bytes", 1_048_576), 1_048_576, minimum=1024)))
            if not isinstance(parsed, expected_type):
                raise ConfigurationTamperingError(str(path), f"{name} must be {expected_type.__name__}", component="cyber_safety")
            return parsed
        except FileNotFoundError:
            if required: raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, f"Required cyber resource missing: {name}", component="cyber_safety")
            return {} if expected_type is dict else []

    def _compile_rule_patterns(self, rules: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        strict = coerce_bool(self._cfg("resource_loading.strict_pattern_validation", True), True)
        for rule in rules or []:
            if not isinstance(rule, Mapping): continue
            for pattern in rule.get("regex_list", []) or []:
                try:
                    output.append({"name": normalize_text(rule.get("name", "unnamed_rule"), max_length=96), "category": normalize_identifier(rule.get("category", "cyber_threat"), max_length=64), "severity": clamp_score(rule.get("severity", 0.5)), "confidence": clamp_score(rule.get("confidence", 0.75)), "pattern": str(pattern), "compiled": re.compile(str(pattern), re.IGNORECASE | re.MULTILINE), "remediation": normalize_text(rule.get("remediation", "Review and mitigate this security finding."), max_length=512), "tags": tuple(str(v) for v in rule.get("tags", []) or [])})
                except re.error as exc:
                    if strict: raise ConfigurationTamperingError("cyber_safety.security_rules", f"Invalid regex: {exc}", component="cyber_safety") from exc
        return output

    def _compile_vulnerability_signatures(self, signatures: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for signature_id, data in (signatures or {}).items():
            if not isinstance(data, Mapping) or not data.get("pattern"): continue
            try:
                output.append({"id": str(signature_id), "name": str(data.get("type", signature_id)), "description": str(data.get("description", data.get("type", "Known vulnerability signature"))), "severity": clamp_score(data.get("severity", 0.8)), "confidence": clamp_score(data.get("confidence", 0.85)), "compiled": re.compile(str(data["pattern"]), re.IGNORECASE | re.MULTILINE | re.DOTALL), "pattern": str(data["pattern"]), "remediation": str(data.get("remediation", "Patch or mitigate the affected component.")), "references": tuple(str(v) for v in data.get("references", []) or [])})
            except re.error as exc:
                if coerce_bool(self._cfg("resource_loading.strict_pattern_validation", True), True):
                    raise ConfigurationTamperingError("cyber_safety.vulnerability_signatures", f"Invalid vulnerability regex: {exc}", component="cyber_safety") from exc
        return output

    def _load_security_model(self) -> Optional[NeuralNetwork]:
        cfg = self._cfg("model", {}) or {}
        if not coerce_bool(cfg.get("enabled", True), True): return None
        path = cfg.get("model_path"); expected = coerce_int(cfg.get("num_inputs", 8), 8, minimum=1)
        if path and Path(str(path)).is_file():
            try:
                model = NeuralNetwork.load_model(str(path), custom_config_override=dict(cfg.get("neural_network_overrides", {}) or {}))
                if model.num_inputs != expected:
                    raise SecurityError(SecurityErrorType.MODEL_TAMPERING, "Cyber-safety model input dimension mismatch.", component="cyber_safety")
                return model
            except SecurityError: raise
            except Exception as exc:
                if coerce_bool(cfg.get("fail_closed_on_prediction_error", False), False):
                    raise wrap_security_exception(exc, operation="load_cyber_safety_model", component="cyber_safety", error_type=SecurityErrorType.MODEL_TAMPERING, severity=SecuritySeverity.HIGH) from exc
                logger.warning("Cyber-safety model unavailable; deterministic degraded mode active.")
        return None

    # ------------------------------------------------------------------
    # Input analysis
    # ------------------------------------------------------------------
    def analyze_input(self, input_data: Any, context: str = "general") -> Dict[str, Any]:
        text = self._stringify_input(input_data)
        context_norm = normalize_identifier(context or "general", max_length=80)
        findings = self._detect_rule_findings(text) + self._detect_vulnerability_findings(text, context_norm) + self._detect_contextual_findings(text, context_norm)
        rule_score = combine_risk_scores(*(f.severity * f.confidence for f in findings if f.finding_type in {"pattern", "heuristic", "secret_exposure"}), method="noisy_or") if findings else 0.0
        vulnerability_score = combine_risk_scores(*(f.severity * f.confidence for f in findings if f.finding_type == "vulnerability_signature"), method="noisy_or") if findings else 0.0
        context_score = self._context_risk_score(context_norm, text)
        model_score = self._score_with_model(text, findings, context_norm)
        values = {"rules": rule_score, "vulnerabilities": vulnerability_score, "context": context_score}
        if model_score is not None: values["model"] = model_score
        risk = weighted_average(values, self._cfg("risk_weights.input", {}) or {}, default=max(values.values() or [0.0]))
        if findings: risk = max(risk, max(f.severity * f.confidence for f in findings))
        risk = clamp_score(risk)
        decision = threshold_decision(risk, block_threshold=coerce_float(self._cfg("risk_thresholds.block", 0.75), 0.75), review_threshold=coerce_float(self._cfg("risk_thresholds.review", 0.5), 0.5))
        result = CyberAnalysisResult(ANALYSIS_SCHEMA_VERSION, generate_identifier("cyber_analysis"), utc_iso(), context_norm, fingerprint(text), risk, categorize_risk(risk), decision, findings, self._generate_recommendations([f.to_dict() for f in findings]), model_score, rule_score, vulnerability_score, context_score, {"input_length": len(text), "finding_count": len(findings), "model_available": self.model is not None, "degraded_mode": self.model is None})
        self._store_analysis("input_analysis", result.to_dict(), coerce_float(self._cfg("memory.input_analysis_sensitivity", 0.75), 0.75))
        return result.to_dict()

    def _stringify_input(self, input_data: Any) -> str:
        if isinstance(input_data, (Mapping, list, tuple, set)):
            return normalize_text(stable_json(input_data), max_length=coerce_int(self._cfg("max_input_text_length", 65536), 65536), preserve_newlines=True)
        return normalize_text(input_data, max_length=coerce_int(self._cfg("max_input_text_length", 65536), 65536), preserve_newlines=True)

    def _detect_rule_findings(self, text: str) -> List[CyberFinding]:
        findings: List[CyberFinding] = []
        for rule in self.compiled_rule_patterns:
            matches = list(rule["compiled"].finditer(text))
            if not matches: continue
            findings.append(self._finding(rule["category"], "pattern", rule["name"], "Configured cyber-security pattern matched.", rule["severity"], rule["confidence"], matches[0].group(0), rule["pattern"], rule["remediation"], rule["tags"], match_count=len(matches)))
        return findings

    def _detect_vulnerability_findings(self, text: str, context: str) -> List[CyberFinding]:
        findings: List[CyberFinding] = []
        for item in self.compiled_vulnerability_patterns:
            matches = list(item["compiled"].finditer(text))
            if matches:
                findings.append(self._finding("vulnerability", "vulnerability_signature", item["name"], item["description"], item["severity"], item["confidence"], matches[0].group(0), item["pattern"], item["remediation"], (), item["references"], len(matches)))
        return findings

    def _detect_contextual_findings(self, text: str, context: str) -> List[CyberFinding]:
        output: List[CyberFinding] = []
        groups = list(self._cfg(f"contextual_heuristics.{context}", []) or []) + list(self._cfg("contextual_heuristics.global", []) or [])
        strict = coerce_bool(self._cfg("resource_loading.strict_pattern_validation", True), True)
        for item in groups:
            if not isinstance(item, Mapping) or not item.get("pattern"): continue
            try: match = re.search(str(item["pattern"]), text, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                if strict: raise ConfigurationTamperingError("cyber_safety.contextual_heuristics", f"Invalid regex: {exc}", component="cyber_safety") from exc
                continue
            if match:
                output.append(self._finding(normalize_identifier(item.get("category", "cyber_threat")), "heuristic", str(item.get("name", "Contextual cyber heuristic")), str(item.get("description", "Contextual cyber-security heuristic matched.")), clamp_score(item.get("severity", 0.5)), clamp_score(item.get("confidence", 0.7)), match.group(0), str(item["pattern"]), str(item.get("remediation", "Review the matched context.")), tuple(str(v) for v in item.get("tags", []) or [])))
        return output

    def _finding(self, category: str, finding_type: str, name: str, description: str, severity: float, confidence: float, evidence: str, pattern: str, remediation: str, tags: Sequence[str] = (), references: Sequence[str] = (), match_count: int = 1) -> CyberFinding:
        return CyberFinding(generate_identifier("cyber_find"), category, finding_type, name, description, severity, confidence, match_count, redact_text(evidence, max_length=160), fingerprint(pattern), remediation, "cyber_safety", tuple(tags), tuple(references))

    def _context_risk_score(self, context: str, text: str) -> float:
        base = self._cfg(f"context_risk.{context}", self._cfg("context_risk.default", 0.0))
        return clamp_score(base)

    def _score_with_model(self, text: str, findings: Sequence[CyberFinding], context: str) -> Optional[float]:
        if self.model is None: return None
        try:
            features = self._build_model_features(text, findings, context)
            prediction = self.model.predict(features)
            return clamp_score(prediction[0] if isinstance(prediction, Sequence) and not isinstance(prediction, (str, bytes)) else prediction)
        except Exception as exc:
            if coerce_bool(self._cfg("model.fail_closed_on_prediction_error", False), False):
                raise wrap_security_exception(exc, operation="score_cyber_input", component="cyber_safety", error_type=SecurityErrorType.UNSAFE_MODEL_STATE, severity=SecuritySeverity.HIGH) from exc
            return None

    def _build_model_features(self, text: str, findings: Sequence[CyberFinding], context: str) -> List[float]:
        cfg = self._cfg("model", {}) or {}; counts: Dict[str, int] = defaultdict(int); max_severity = 0.0
        for finding in findings: counts[finding.finding_type] += finding.match_count; max_severity = max(max_severity, finding.severity)
        feature_map = {
            "length": clamp_score(len(text) / coerce_int(self._cfg("input_size_limit", 65536), 65536, minimum=1)),
            "entropy": self._normalized_entropy(text), "finding_count": clamp_score(len(findings) / coerce_float(cfg.get("finding_count_scale", 10.0), 10.0, minimum=1.0)),
            "max_severity": max_severity, "secret_exposure_count": clamp_score(counts.get("secret_exposure", 0) / 5.0),
            "injection_count": clamp_score(counts.get("pattern", 0) / 5.0), "vulnerability_count": clamp_score(counts.get("vulnerability_signature", 0) / 3.0),
            "context_risk": self._context_risk_score(context, text),
        }
        order = list(cfg.get("feature_order", [])) or list(feature_map)
        expected = coerce_int(cfg.get("num_inputs", 8), 8, minimum=1)
        values = [coerce_float(feature_map.get(name, 0.0), 0.0, minimum=0.0, maximum=1.0) for name in order]
        values.extend([0.0] * max(0, expected - len(values)))
        return values[:expected]

    def _normalized_entropy(self, text: str) -> float:
        if not text: return 0.0
        counts: Dict[str, int] = defaultdict(int)
        for char in text: counts[char] += 1
        total = float(len(text)); entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
        return clamp_score(entropy / 8.0)

    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        values = [str(f.get("remediation")) for f in findings if f.get("remediation")]
        if not values: values.append("No configured cyber-security concern was detected; continue routine monitoring.")
        return dedupe_preserve_order(values)[:coerce_int(self._cfg("detection.max_recommendations", 12), 12, minimum=1)]

    # ------------------------------------------------------------------
    # Event-stream anomaly detection
    # ------------------------------------------------------------------
    def analyze_event_stream(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(event, Mapping) or "timestamp" not in event:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Event stream item must be a mapping with timestamp.", component="cyber_safety")
        safe_event = sanitize_for_logging(dict(event)); event_type = normalize_identifier(event.get("type", "unknown_event"), max_length=96)
        entity = normalize_text(event.get("user") or event.get("ip_address") or event.get("actor_id") or "global", max_length=128)
        entity_class = normalize_identifier(event.get("entity_class") or event.get("source_type") or "default", max_length=64)
        population = f"{event_type}:{entity_class}"
        details = event.get("details", {}) if isinstance(event.get("details", {}), Mapping) else {}
        numerical = {str(k): coerce_float(v, 0.0) for k, v in details.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        feature_scores: Dict[str, float] = {}; contributors: List[str] = []; min_samples = coerce_int(self._cfg("event_stream.min_stat_samples", 30), 30, minimum=2)
        with self._state_lock:
            self.event_log_history.append(safe_event)
            self._event_counts[population] += 1
            # Score against prior baseline BEFORE learning this event.
            pending_updates: List[Tuple[RunningStatistic, float]] = []
            for feature, value in numerical.items():
                stats = self.event_statistics[f"{population}:{normalize_identifier(feature, max_length=64)}"]
                z = stats.z_score(value) if stats.count >= min_samples else 0.0
                if z > 0: feature_scores[f"stat:{feature}"] = z
                if z >= self.anomaly_threshold: contributors.append(f"statistical:{feature}")
                pending_updates.append((stats, value))
            sequence_score = self._score_sequence_locked(entity, event_type, population)
            if sequence_score > 0: feature_scores["sequence"] = sequence_score
            if sequence_score >= self.anomaly_threshold: contributors.append("sequence")
            centroid_score = 0.0
            vector = self._map_to_feature_vector(numerical) if self.use_centroid and numerical else None
            if vector is not None and np.any(vector):
                centroid_score = self._centroid_anomaly_locked(population, vector, update=False)
                feature_scores["adaptive_centroid"] = centroid_score
                if centroid_score >= self.anomaly_threshold: contributors.append("adaptive_centroid")
            final_raw = max(feature_scores.values()) if feature_scores else 0.0
            is_anomaly = final_raw >= self.anomaly_threshold
            if not is_anomaly or coerce_bool(self._cfg("event_stream.learn_from_anomalies", False), False):
                for stats, value in pending_updates: stats.update(value)
                if vector is not None and np.any(vector): self._centroid_anomaly_locked(population, vector, update=True)
            sequence = tuple(self.sequence_patterns[f"seq:{fingerprint(entity)}"])
        normalized_score = clamp_score(final_raw / max(self.anomaly_threshold * 2.0, 1.0))
        decision = threshold_decision(normalized_score, block_threshold=coerce_float(self._cfg("risk_thresholds.event_block", 0.85), 0.85), review_threshold=coerce_float(self._cfg("risk_thresholds.event_review", 0.5), 0.5))
        reason = "Within learned/reference parameters." if not is_anomaly else f"Anomaly score {final_raw:.3f} exceeded threshold {self.anomaly_threshold:.3f}."
        result = EventAnalysisResult(EVENT_SCHEMA_VERSION, generate_identifier("cyber_event"), utc_iso(), event_type, entity, round(final_raw, 6), normalized_score, is_anomaly, categorize_risk(normalized_score), decision, reason, dedupe_preserve_order(contributors), {k: round(v, 6) for k, v in feature_scores.items()}, sequence, {"population": fingerprint(population), "numeric_feature_count": len(numerical), "event_fingerprint": fingerprint(safe_event)})
        self._store_analysis("event_analysis", result.to_dict(), coerce_float(self._cfg("memory.event_analysis_sensitivity", 0.65), 0.65))
        return result.to_dict()

    def _score_sequence_locked(self, entity: str, event_type: str, population: str) -> float:
        key = f"seq:{fingerprint(entity)}"; seq = self.sequence_patterns[key]; seq.append(event_type)
        min_len = coerce_int(self._cfg("event_stream.min_sequence_length", 3), 3, minimum=2)
        warmup = coerce_int(self._cfg("event_stream.sequence_warmup_events", 30), 30, minimum=0)
        if len(seq) < min_len or self._event_counts[population] <= warmup: return 0.0
        sequence_key = fingerprint((population, tuple(seq))); count = self.sequence_counts[sequence_key]; self.sequence_counts[sequence_key] = count + 1
        weight = coerce_float(self._cfg("event_stream.sequence_novelty_weight", 5.0), 5.0, minimum=0.0)
        # Cold novelty is damped by +1 pseudo-count and capped to the statistical scale.
        return min(self.anomaly_threshold * 2.0, max(0.0, weight - math.log1p(count + 1)))

    def _map_to_feature_vector(self, numerical: Mapping[str, Any]) -> np.ndarray:
        vector = np.zeros(self.qnn_feature_dim, dtype=float); scale = coerce_float(self._cfg("event_stream.feature_scale", 1000.0), 1000.0, minimum=1.0)
        for name in sorted(str(k) for k in numerical):
            value = coerce_float(numerical[name], 0.0); idx = int(hash_text(name), 16) % self.qnn_feature_dim; vector[idx] += math.tanh(value/scale) if abs(value) > 10 else value
        norm = np.linalg.norm(vector); return vector / norm if norm > 1e-9 else vector

    def _centroid_anomaly_locked(self, population: str, vector: np.ndarray, *, update: bool) -> float:
        centroid = self.centroids.get(population)
        if centroid is None:
            if update: self.centroids[population] = vector.copy()
            return 0.0
        c_norm = np.linalg.norm(centroid); v_norm = np.linalg.norm(vector)
        if c_norm < 1e-9 or v_norm < 1e-9: return 0.0
        c = centroid / c_norm; v = vector / v_norm; distance = 1.0 - float(np.clip(np.dot(v, c), -1.0, 1.0))
        if update:
            updated = (1.0 - self.adaptive_centroid_lr) * c + self.adaptive_centroid_lr * v; norm = np.linalg.norm(updated); self.centroids[population] = updated / norm if norm > 1e-9 else updated
        return (distance ** 2) * coerce_float(self._cfg("event_stream.qnn_distance_scale", 2.5), 2.5, minimum=0.1)

    # ------------------------------------------------------------------
    # Threat modeling
    # ------------------------------------------------------------------
    def generate_threat_assessment(self, context_info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(context_info, Mapping):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Threat assessment context must be a mapping.", component="cyber_safety")
        component = normalize_text(context_info.get("component", "unknown_component"), max_length=128, lowercase=True); action = normalize_text(context_info.get("action", "unknown_action"), max_length=128, lowercase=True); zone = normalize_text(context_info.get("network_zone", "internal"), max_length=80, lowercase=True)
        data_involved = context_info.get("data_involved", []); context_blob = stable_json(sanitize_for_logging(context_info)); threats: Dict[str, List[str]] = defaultdict(list)
        for rule in self._cfg("threat_modeling.rules", []) or []:
            if not isinstance(rule, Mapping): continue
            values = {"component": component, "action": action, "network_zone": zone, "data": stable_json(data_involved), "context": context_blob}; matched = False
            for field_name, value in values.items():
                for pattern in rule.get(f"{field_name}_patterns", []) or []:
                    try:
                        if re.search(str(pattern), str(value), re.IGNORECASE): matched = True; break
                    except re.error as exc: raise ConfigurationTamperingError("cyber_safety.threat_modeling.rules", f"Invalid regex: {exc}", component="cyber_safety") from exc
                if matched: break
            if matched: threats[str(rule.get("category", "Unknown"))].append(str(rule.get("description", "Potential cyber threat.")))
        weights = self._cfg("threat_modeling.category_weights", {}) or {}; parts = {category: clamp_score(coerce_float(weights.get(category, 0.4), 0.4) + min(len(items), 5) * 0.06) for category, items in threats.items()}
        risk = combine_risk_scores(*parts.values(), method="noisy_or") if parts else 0.0
        result = ThreatAssessmentResult(THREAT_SCHEMA_VERSION, generate_identifier("threat"), utc_iso(), component, action, zone, risk, categorize_risk(risk).title(), dict(threats), self._threat_mitigations(threats), {"threat_count": sum(len(v) for v in threats.values()), "context_fingerprint": fingerprint(context_blob)})
        self._store_analysis("threat_assessment", result.to_dict(), coerce_float(self._cfg("memory.threat_assessment_sensitivity", 0.55), 0.55))
        return result.to_dict()

    def _threat_mitigations(self, threats: Mapping[str, List[str]]) -> List[str]:
        mapping = self._cfg("threat_modeling.mitigations", {}) or {}; values: List[str] = []
        for category in threats:
            configured = mapping.get(category, [])
            values.extend([configured] if isinstance(configured, str) else [str(v) for v in configured])
        return dedupe_preserve_order(values or (["Perform a focused threat-model review and document compensating controls."] if threats else ["No STRIDE threats matched; continue routine least-privilege monitoring."]))

    def generate_report(self, analysis: Mapping[str, Any]) -> str:
        safe = sanitize_for_logging(dict(analysis or {}))
        return "\n".join(["# Cyber Safety Report", f"**Generated**: {utc_iso()}", f"**Risk Score**: {coerce_float(safe.get('risk_score', safe.get('normalized_anomaly_score', 0.0)), 0.0):.3f}", f"**Decision**: {safe.get('decision', 'unknown')}", "", "```json", stable_json(safe), "```"])

    def _store_analysis(self, tag: str, payload: Mapping[str, Any], sensitivity: float) -> None:
        if not coerce_bool(self._cfg("memory.store_analysis", True), True): return
        try:
            self.memory.add(sanitize_for_logging(dict(payload)), tags=["cyber_safety", "analysis", tag], sensitivity=sensitivity, ttl_seconds=coerce_int(self._cfg("memory.analysis_ttl_seconds", 86400), 86400, minimum=0), purpose="cyber_safety_analysis", owner="cyber_safety", source="cyber_safety", metadata={"eligible_for_compliance": True, "analysis_fingerprint": fingerprint(payload)})
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_store_error", False), False):
                raise AuditLogFailureError("secure_memory.cyber_safety", f"Failed to store cyber analysis: {type(exc).__name__}", component="cyber_safety", cause=exc) from exc


__all__ = ["MODULE_VERSION", "ANALYSIS_SCHEMA_VERSION", "EVENT_SCHEMA_VERSION", "THREAT_SCHEMA_VERSION", "CyberFinding", "CyberAnalysisResult", "EventAnalysisResult", "RunningStatistic", "ThreatAssessmentResult", "CyberSafetyModule"]
