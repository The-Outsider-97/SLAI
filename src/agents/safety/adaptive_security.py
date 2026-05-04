"""
Adaptive phishing and cyber-security orchestration for the Safety Agent subsystem.

This module coordinates SafetyFeatures, NeuralNetwork, SecureMemory, and the
shared security helper/error layers to assess emails, URLs, traffic payloads, and
supply-chain artifacts for phishing and related cyber risks.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import time

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from .utils.config_loader import load_global_config, get_config_section
from .utils.safety_helpers import *
from .utils.security_error import *
from .modules.neural_network import NeuralNetwork
from .modules.safety_features import SafetyFeatures
from .secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Adaptive Security System")
printer = PrettyPrinter()

MODULE_VERSION = "2.1.0"
ANALYSIS_SCHEMA_VERSION = "adaptive_security.analysis.v3"
SUPPLY_CHAIN_SCHEMA_VERSION = "adaptive_security.supply_chain.v2"


@dataclass(frozen=True)
class AdaptiveAnalysisResult:
    """Audit-safe analysis envelope returned by email and URL analysis."""

    source_type: str
    phishing_score: float
    is_phishing: bool
    decision: str
    risk_level: str
    threat_type: Optional[str]
    features: List[float]
    model_score: float
    heuristic_score: float
    confidence: float
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = ANALYSIS_SCHEMA_VERSION
        data["timestamp"] = self.timestamp or utc_iso()
        data["metadata"] = sanitize_for_logging(data.get("metadata", {}))
        data["features"] = [float(v) for v in data.get("features", [])]
        return data


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of per-principal rate-limit evaluation."""

    allowed: bool
    principal: str
    count: int
    limit: int
    window_seconds: int
    retry_after_seconds: float = 0.0
    reason: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging(asdict(self))


@dataclass(frozen=True)
class SupplyChainCheckResult:
    """Audit-safe result for trusted-hash verification."""

    file_fingerprint: str
    file_hash: str
    is_trusted: bool
    risk_level: str
    decision: str
    matched_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SUPPLY_CHAIN_SCHEMA_VERSION
        data["timestamp"] = self.timestamp or utc_iso()
        data["metadata"] = sanitize_for_logging(data.get("metadata", {}))
        return data


class AdaptiveSecurity:
    """
    Comprehensive cyber-security system focused on phishing detection and threat
    prevention.

    This class combines neural-network scoring with deterministic security
    features and policy thresholds. It does not own low-level helper logic,
    persistence internals, or model implementation details.
    """

    _IPV4_PATTERN = re.compile(
        r"\b(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    _IPV6_PATTERN = re.compile(
        r"\b(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}\b|"
        r"\b(?:[A-F0-9]{1,4}:){1,7}:\b|"
        r"\b(?:[A-F0-9]{1,4}:){1,6}:[A-F0-9]{1,4}\b|"
        r"\b::(?:[A-F0-9]{1,4}:){0,5}[A-F0-9]{1,4}\b|"
        r"\b(?:[A-F0-9]{1,4}:){1,5}::\b",
        re.IGNORECASE,
    )
    _IP_WITH_PORT = re.compile(r":\d{1,5}\b")
    _IP_IN_URL = re.compile(r"(?i)\bhttps?://([^/\s?#]+)")

    def __init__(self):
        self.config = load_global_config()
        self.adaptive_config = get_config_section("adaptive_security")
        self._validate_configuration()

        self.safety_features = SafetyFeatures()
        self.memory = SecureMemory()

        self.rate_limit = coerce_int(self._cfg("rate_limit"), 30, minimum=1)
        self.input_size_limit = coerce_int(self._cfg("input_size_limit"), 65536, minimum=1)
        self.phishing_threshold = clamp_score(self._cfg("phishing_threshold"))
        self.review_threshold = clamp_score(self._cfg("review_threshold"))
        self.block_threshold = clamp_score(self._cfg("block_threshold"))
        self.email_model_path = str(self._cfg("email_model_path"))
        self.url_model_path = str(self._cfg("url_model_path"))

        rate_limit_maxlen = coerce_int(self._cfg("rate_limit_tracker_maxlen"), 200, minimum=1)
        self.request_tracker: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=rate_limit_maxlen))
        self.blocked_ips: Dict[str, Dict[str, Any]] = {}
        self.safe_package_hashes = self._load_trusted_hashes()
        self.last_email_content = ""

        self.email_nn = self._initialize_neural_network("email")
        self.url_nn = self._initialize_neural_network("url")

        logger.info("AdaptiveSecurity initialized: %s", safe_log_payload(
            "adaptive_security_initialized",
            {
                "schema_version": self._cfg("schema_version"),
                "rate_limit": self.rate_limit,
                "phishing_threshold": self.phishing_threshold,
                "trusted_hash_count": len(self.safe_package_hashes),
                "email_model_path": self.email_model_path,
                "url_model_path": self.url_model_path,
            },
        ))

    def _add_memory_entry(self, data: Any, tags: List[str], sensitivity: float, ttl_seconds: Optional[int] = None,
                          purpose: str = "adaptive_security", source: str = "adaptive_security",
                          metadata: Optional[Mapping[str, Any]] = None) -> str:
        """Add an entry to secure memory with audit context."""
        return self.memory.add(
            entry=data,
            tags=tags,
            sensitivity=sensitivity,
            ttl_seconds=ttl_seconds,
            purpose=purpose,
            owner="adaptive_security",
            source=source,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.adaptive_config, path, default)

    def _validate_configuration(self) -> None:
        require_keys(
            self.adaptive_config,
            [
                "rate_limit",
                "rate_limit_window_seconds",
                "input_size_limit",
                "phishing_threshold",
                "review_threshold",
                "block_threshold",
                "email_model_path",
                "url_model_path",
                "email_layer_config",
                "url_layer_config",
                "feature_schema",
                "score_weights",
                "trusted_hashes",
                "supply_chain",
                "training",
                "memory",
            ],
            context="adaptive_security",
        )
    
        if clamp_score(self.adaptive_config.get("review_threshold")) > clamp_score(self.adaptive_config.get("block_threshold")):
            raise ConfigurationTamperingError(
                "adaptive_security.review_threshold",
                "review_threshold must be less than or equal to block_threshold",
                component="adaptive_security",
            )
    
        for model_type in ("email", "url"):
            layers = self._cfg(f"{model_type}_layer_config")
            if not isinstance(layers, Sequence) or not layers:
                raise ConfigurationTamperingError(
                    f"adaptive_security.{model_type}_layer_config",
                    "model layer config must be a non-empty sequence",
                    component="adaptive_security",
                )
    
        weights = self._cfg("score_weights")
        if not isinstance(weights, Mapping) or not weights:
            raise ConfigurationTamperingError(
                "adaptive_security.score_weights",
                "score_weights must be a mapping",
                component="adaptive_security",
            )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _model_settings(self, model_type: str) -> Dict[str, Any]:
        if model_type not in {"email", "url"}:
            raise ConfigurationTamperingError(
                "adaptive_security.model_type",
                f"Unsupported model type: {model_type}",
                component="adaptive_security",
            )
        schema = self._cfg("feature_schema", {})
        return {
            "model_type": model_type,
            "num_inputs": coerce_int(schema.get(f"{model_type}_num_inputs"), 0, minimum=1),
            "model_path": str(self._cfg(f"{model_type}_model_path")),
            "layer_config": list(self._cfg(f"{model_type}_layer_config")),
            "loss_function_name": str(self._cfg("model_defaults.loss_function_name")),
            "optimizer_name": str(self._cfg("model_defaults.optimizer_name")),
            "problem_type": str(self._cfg("model_defaults.problem_type")),
        }

    def _initialize_neural_network(self, model_type: str) -> NeuralNetwork:
        settings = self._model_settings(model_type)
        model_path = settings["model_path"]
        allow_fallback = coerce_bool(self._cfg("model_loading.allow_untrained_fallback"), True)
        verify_on_load = coerce_bool(self._cfg("model_loading.verify_on_load"), True)

        if model_path and Path(model_path).exists():
            try:
                logger.info("Loading %s phishing model: %s", model_type, sanitize_for_logging({"path": model_path}))
                loader_kwargs = {}
                try:
                    code_obj = getattr(NeuralNetwork.load_model, "__code__", None)
                    if code_obj and "custom_config_override" in code_obj.co_varnames:
                        loader_kwargs["custom_config_override"] = get_config_section("neural_network")
                except AttributeError:
                    loader_kwargs = {}
                loaded_model = NeuralNetwork.load_model(model_path, **loader_kwargs)
                expected_inputs = settings["num_inputs"]
                actual_inputs = getattr(loaded_model, "num_inputs", expected_inputs)
                if actual_inputs != expected_inputs:
                    raise ModelTamperingDetectedError(
                        model_name=model_path,
                        detection_method="adaptive_security_input_dimension_check",
                        expected_hash=str(expected_inputs),
                        actual_hash=str(actual_inputs),
                        component="adaptive_security",
                    )
                if verify_on_load:
                    self._audit_model_load(model_type, model_path, loaded_model)
                return loaded_model
            except SecurityError:
                if not allow_fallback:
                    raise
                logger.warning("Model load failed; falling back to configured architecture: %s", safe_log_payload(
                    "adaptive_model_load_failed",
                    {"model_type": model_type, "path": model_path},
                ))
            except Exception as exc:
                wrapped = wrap_security_exception(
                    exc,
                    operation=f"load_{model_type}_model",
                    component="adaptive_security",
                    context={"model_type": model_type, "path": model_path},
                    error_type=SecurityErrorType.MODEL_TAMPERING,
                    severity=SecuritySeverity.HIGH,
                )
                if not allow_fallback:
                    raise wrapped from exc
                logger.warning("Model load failed; fallback enabled: %s", wrapped.to_log_record() if hasattr(wrapped, "to_log_record") else str(wrapped))

        if not allow_fallback and model_path:
            raise SystemIntegrityError(
                component="adaptive_security.model_loading",
                anomaly_description=f"Configured {model_type} model path does not exist and fallback is disabled.",
                expected_state=model_path,
                actual_state="missing",
            )

        return NeuralNetwork(
            num_inputs=settings["num_inputs"],
            layer_config=settings["layer_config"],
            loss_function_name=settings["loss_function_name"],
            optimizer_name=settings["optimizer_name"],
            problem_type=settings["problem_type"],
            config=get_config_section("neural_network"),
        )

    def _audit_model_load(self, model_type: str, model_path: str, model: NeuralNetwork) -> None:
        if not coerce_bool(self._cfg("memory.store_model_events"), True):
            return
        try:
            self._add_memory_entry(
                {
                    "event": "adaptive_model_loaded",
                    "model_type": model_type,
                    "model_path_fingerprint": fingerprint(model_path),
                    "num_inputs": getattr(model, "num_inputs", None),
                    "model_fingerprint": fingerprint(model.get_weights_biases()) if hasattr(model, "get_weights_biases") else None,
                    "timestamp": utc_iso(),
                },
                tags=list(self._cfg("memory.model_event_tags", ["adaptive_security", "model"])),
                sensitivity=coerce_float(self._cfg("memory.model_event_sensitivity"), 0.65, minimum=0.0, maximum=1.0),
                purpose="adaptive_security_model_governance",
                source="adaptive_security",
            )
        except Exception as exc:
            raise AuditLogFailureError(
                logging_target="secure_memory.model_events",
                failure_mode=type(exc).__name__,
                component="adaptive_security",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def analyze_email(
        self,
        email: Mapping[str, Any],
        *,
        client_ip: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze email for phishing characteristics."""

        printer.status("ADAPT", "Analyzing email", "info")
        if not isinstance(email, Mapping):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Email analysis requires a mapping payload.",
                severity=SecuritySeverity.HIGH,
                context={"payload_type": type(email).__name__},
                component="adaptive_security",
            )

        self._validate_payload_size(email, source_type="email")
        self._remember_last_email(email)
        features = self._extract_email_features(email)
        feature_result = self.safety_features.assess_email_risk(email) if hasattr(self.safety_features, "assess_email_risk") else None
        return self._analyze_features(features, "email", self.email_nn, client_ip=client_ip, context=context, feature_result=feature_result)

    def analyze_url(
        self,
        url: str,
        *,
        client_ip: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze URL for phishing characteristics."""

        printer.status("ADAPT", "Analyzing URL", "info")
        normalized_url = normalize_text(url, max_length=coerce_int(self._cfg("max_url_length"), 2048))
        if not normalized_url:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "URL analysis requires a non-empty URL.",
                severity=SecuritySeverity.HIGH,
                context={"url_fingerprint": fingerprint(url)},
                component="adaptive_security",
            )

        self._validate_payload_size(normalized_url, source_type="url")
        features = self._extract_url_features(normalized_url)
        feature_result = self.safety_features.assess_url_risk(normalized_url) if hasattr(self.safety_features, "assess_url_risk") else None
        return self._analyze_features(features, "url", self.url_nn, client_ip=client_ip, context=context, feature_result=feature_result)

    def _analyze_features(
        self,
        features: Sequence[float],
        source_type: str,
        model: NeuralNetwork,
        *,
        client_ip: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        feature_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Common analysis path for email and URL feature vectors."""

        printer.status("ADAPT", "Analyzing features", "info")
        feature_vector = self._validate_features(features, source_type)
        rate_decision = self._rate_limit_decision(client_ip or self._get_client_ip())
        if not rate_decision.allowed:
            return self._blocked_rate_limit_result(source_type, feature_vector, rate_decision)

        try:
            raw_prediction = model.predict(feature_vector)
            model_score = clamp_score(raw_prediction[0] if isinstance(raw_prediction, Sequence) else raw_prediction)
        except SecurityError:
            raise
        except Exception as exc:
            raise wrap_security_exception(
                exc,
                operation=f"predict_{source_type}_phishing_score",
                component="adaptive_security",
                context={"source_type": source_type, "feature_fingerprint": fingerprint(feature_vector)},
                error_type=SecurityErrorType.UNSAFE_MODEL_STATE,
                severity=SecuritySeverity.CRITICAL,
            ) from exc

        heuristic_score, indicators = self._heuristic_score(source_type, feature_vector, feature_result)
        combined_score = weighted_average(
            {"model": model_score, "heuristic": heuristic_score},
            self._cfg("score_weights", {}),
            default=model_score,
        )
        threat_type = self._determine_threat_type(feature_vector, source_type, indicators=indicators)
        decision = threshold_decision(combined_score, block_threshold=self.block_threshold, review_threshold=self.review_threshold)
        risk_level = categorize_risk(combined_score)
        is_phishing = bool(combined_score >= self.phishing_threshold or decision == "block")

        result = AdaptiveAnalysisResult(
            source_type=source_type,
            phishing_score=combined_score,
            is_phishing=is_phishing,
            decision=decision,
            risk_level=risk_level,
            threat_type=threat_type,
            features=list(feature_vector),
            model_score=model_score,
            heuristic_score=heuristic_score,
            confidence=self._confidence_score(model_score, heuristic_score),
            indicators=indicators,
            metadata={
                "rate_limit": rate_decision.to_dict(),
                "context": sanitize_for_logging(context or {}),
                "feature_result": feature_result.to_dict() if feature_result is not None and hasattr(feature_result, "to_dict") else sanitize_for_logging(feature_result),
                "feature_fingerprint": fingerprint(feature_vector),
            },
            timestamp=utc_iso(),
        )
        self._store_analysis_result(result)
        logger.info("Adaptive analysis completed: %s", safe_log_payload("adaptive_analysis_completed", result.to_dict()))
        return result.to_dict()

    def _blocked_rate_limit_result(self, source_type: str, features: Sequence[float], decision: RateLimitDecision) -> Dict[str, Any]:
        result = AdaptiveAnalysisResult(
            source_type=source_type,
            phishing_score=1.0,
            is_phishing=True,
            decision="block",
            risk_level="critical",
            threat_type="Resource Exhaustion / Rate Limit Abuse",
            features=list(features),
            model_score=0.0,
            heuristic_score=1.0,
            confidence=1.0,
            indicators=["rate_limit_exceeded"],
            metadata={"rate_limit": decision.to_dict()},
            timestamp=utc_iso(),
        )
        self._store_analysis_result(result)
        return result.to_dict()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_email_features(self, email: Mapping[str, Any]) -> List[float]:
        """Convert email characteristics to the configured 11-feature vector."""

        printer.status("ADAPT", "Extracting email features", "info")
        if hasattr(self.safety_features, "extract_email_feature_vector"):
            features = self.safety_features.extract_email_feature_vector(email)
        else:
            features = [
                len(str(email.get("from", ""))),
                len(str(email.get("subject", ""))),
                self.safety_features._contains_suspicious_keywords(str(email.get("subject", ""))),
                len(email.get("links", []) or []),
                self.safety_features._contains_urgent_language(str(email.get("body", ""))),
                self.safety_features._contains_attachment(dict(email)),
                self.safety_features._domain_mismatch_score(dict(email)),
                self.safety_features._avg_url_length(email.get("links", []) or []),
                self.safety_features._ssl_cert_score(email.get("links", []) or []),
                self.safety_features._unusual_sender_score(email.get("from", "")),
                self.safety_features._unusual_time_score(email.get("timestamp", 0)),
            ]
        return self._validate_features(features, "email")

    def _extract_url_features(self, url: str) -> List[float]:
        """Convert URL characteristics to the configured 8-feature vector."""

        printer.status("ADAPT", "Extracting URL features", "info")
        if hasattr(self.safety_features, "extract_url_feature_vector"):
            features = self.safety_features.extract_url_feature_vector(url)
        else:
            ip_result = self._contains_ip_address(url, validate=True)
            contains_ip = ip_result[0] if isinstance(ip_result, tuple) else ip_result
            features = [
                float(len(url)),
                float(self.safety_features._url_entropy(url)),
                float(self.safety_features._num_subdomains(url)),
                float(contains_ip),
                float(self.safety_features._https_used(url)),
                float(self.safety_features._url_redirect_count(url)),
                float(self._domain_age(url)),
                float(self.safety_features._special_char_count(url)),
            ]
        return self._validate_features(features, "url")

    def _validate_features(self, features: Sequence[Any], source_type: str) -> List[float]:
        schema = self._cfg("feature_schema", {})
        expected = coerce_int(schema.get(f"{source_type}_num_inputs"), 0, minimum=1)
        if len(features) != expected:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                f"{source_type} feature vector has invalid length.",
                severity=SecuritySeverity.HIGH,
                context={"source_type": source_type, "expected": expected, "actual": len(features)},
                component="adaptive_security",
            )
        clean: List[float] = []
        for idx, value in enumerate(features):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise SecurityError(
                    SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                    f"Non-numeric {source_type} feature encountered.",
                    severity=SecuritySeverity.HIGH,
                    context={"source_type": source_type, "feature_index": idx, "value_repr": safe_repr(value)},
                    component="adaptive_security",
                    cause=exc,
                ) from exc
            if not math.isfinite(numeric):
                raise SecurityError(
                    SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                    f"Non-finite {source_type} feature encountered.",
                    severity=SecuritySeverity.HIGH,
                    context={"source_type": source_type, "feature_index": idx},
                    component="adaptive_security",
                )
            clean.append(numeric)
        return clean

    def _validate_payload_size(self, payload: Any, *, source_type: str) -> None:
        size = len(stable_json(payload).encode("utf-8", errors="replace"))
        limit = coerce_int(self._cfg("input_size_limit"), 65536, minimum=1)
        if size > limit:
            raise ResourceExhaustionError(
                resource_type=f"{source_type}_payload_bytes",
                current_usage=float(size),
                limit=float(limit),
                source_identifier=source_type,
                component="adaptive_security",
            )

    # ------------------------------------------------------------------
    # Heuristics and threat typing
    # ------------------------------------------------------------------

    def _heuristic_score(self, source_type: str, features: Sequence[float], feature_result: Optional[Any]) -> Tuple[float, List[str]]:
        indicators: List[str] = []
        if feature_result is not None:
            risk_score = clamp_score(getattr(feature_result, "risk_score", 0.0) if not isinstance(feature_result, Mapping) else feature_result.get("risk_score", 0.0))
            raw_indicators = getattr(feature_result, "indicators", None) if not isinstance(feature_result, Mapping) else feature_result.get("indicators", [])
            indicators.extend(str(item) for item in (raw_indicators or []))
            return risk_score, dedupe_preserve_order(indicators)

        if source_type == "email":
            scores = {
                "suspicious_keywords": features[2],
                "urgent_language": features[4],
                "attachment": features[5],
                "domain_mismatch": features[6],
                "unusual_sender": features[9],
                "unusual_time": features[10],
            }
            weights = self._cfg("email_heuristic_weights", {})
        else:
            scores = {
                "entropy": clamp_score(features[1]),
                "subdomains": clamp_score(features[2] / 5.0),
                "contains_ip": features[3],
                "missing_https": 1.0 - clamp_score(features[4]),
                "redirects": clamp_score(features[5]),
                "domain_age": features[6],
                "special_chars": clamp_score(features[7]),
            }
            weights = self._cfg("url_heuristic_weights", {})

        for name, value in scores.items():
            indicator_threshold = clamp_score(self._cfg("indicator_threshold", 0.65))
            if clamp_score(value) >= indicator_threshold:
                indicators.append(name)
        return weighted_average(scores, weights, default=0.0), dedupe_preserve_order(indicators)

    def _confidence_score(self, model_score: float, heuristic_score: float) -> float:
        agreement = 1.0 - abs(clamp_score(model_score) - clamp_score(heuristic_score))
        strength = max(abs(model_score - 0.5), abs(heuristic_score - 0.5)) * 2.0
        return clamp_score((agreement * 0.55) + (strength * 0.45))

    def _determine_threat_type(self, features: Sequence[float], source_type: str, *, indicators: Optional[Sequence[str]] = None) -> Optional[str]:
        """Classify likely threat type using feature pattern combinations."""

        printer.status("ADAPT", "Determining threat type", "info")
        indicators = list(indicators or [])

        if source_type == "email":
            if features[6] >= 0.85 and features[9] >= 0.75:
                return "Phishing: Targeted Impersonation Attack"
            if features[6] >= 0.85:
                return "Phishing: Domain Impersonation"
            if features[5] >= 0.9 and features[3] == 0:
                return "Malware: Suspicious Attachment Delivery"
            if features[3] >= 3 and features[7] >= coerce_float(self._cfg("threat_thresholds.email.long_url_length"), 60.0):
                return "Phishing: Credential Harvesting"
            if features[4] >= 0.85 and features[2] >= 0.65:
                if self._last_email_has_financial_brand():
                    return "Financial Scam: Payment Service Impersonation"
                return "Phishing: Urgent Social Engineering"
            if features[10] >= 0.95:
                return "Phishing: Temporal Anomaly Attack"
            if self._detect_advanced_threat_pattern(list(features), source_type):
                return "Advanced Persistent Threat Pattern"
            if "attachment_risk" in indicators:
                return "Malware: Attachment Risk"
            if "sender_reputation" in indicators or "domain_mismatch" in indicators:
                return "Phishing: Sender Reputation Anomaly"

        if source_type == "url":
            if features[3] >= 0.9 and features[4] < 0.5:
                return "Phishing: Direct IP Access (Insecure)"
            if features[3] >= 0.9:
                return "Phishing: Suspicious IP URL"
            if features[1] >= coerce_float(self._cfg("threat_thresholds.url.high_entropy"), 0.65) and features[2] >= 4:
                return "Phishing: Multi-layer URL Obfuscation"
            if features[6] >= 0.85 and features[5] >= 0.5:
                return "Phishing: New Domain Redirect Chain"
            if features[5] >= 0.75:
                return "Phishing: Multi-hop Redirect"
            if features[4] >= 0.9 and features[1] >= coerce_float(self._cfg("threat_thresholds.url.high_entropy"), 0.65):
                return "Phishing: HTTPS Spoofing"
            if "credential_in_url" in indicators:
                return "Credential Exposure: URL-embedded Credentials"
            if "punycode_or_homograph" in indicators:
                return "Phishing: Homograph / Punycode Attack"
            if self._detect_advanced_threat_pattern(list(features), source_type):
                return "Advanced Persistent Threat Pattern"

        return None

    def _detect_advanced_threat_pattern(self, features: List[float], source_type: str) -> bool:
        if source_type == "email":
            risk_score = (
                0.30 * clamp_score(features[6]) +
                0.20 * clamp_score(features[9]) +
                0.25 * clamp_score(features[4]) +
                0.15 * clamp_score(features[2]) +
                0.10 * clamp_score(features[10])
            )
            apt_threshold = clamp_score(self._cfg("threat_thresholds.email.apt_score", 0.65))
            return risk_score > apt_threshold and features[3] > 1

        if source_type == "url":
            return (
                clamp_score(features[3]) > 0.70 and
                clamp_score(features[6]) > 0.60 and
                clamp_score(features[1]) > clamp_score(self._cfg("threat_thresholds.url.apt_entropy"), 0.55) and # type: ignore
                clamp_score(features[5]) > 0.20
            )
        return False

    def _remember_last_email(self, email: Mapping[str, Any]) -> None:
        content = f"{email.get('from', '')} {email.get('subject', '')} {email.get('body', '')}"
        self.last_email_content = normalize_text(content, max_length=coerce_int(self._cfg("max_text_length"), 4096), lowercase=True)

    def _last_email_has_financial_brand(self) -> bool:
        brands = [normalize_text(item, lowercase=True) for item in self._cfg("financial_brand_indicators", [])]
        return any(brand and brand in self.last_email_content for brand in brands)

    # ------------------------------------------------------------------
    # Domain, IP, and network helpers
    # ------------------------------------------------------------------

    def _domain_age(self, url: str) -> float:
        """Return suspiciousness score for domain age: 1.0 means newer/riskier."""

        printer.status("ADAPT", "Calculating domain age", "info")
        domain = self._extract_domain(url)
        if not domain:
            return clamp_score(self._cfg("domain_age.unknown_score"))

        cache_key = f"domain_age:{domain}"
        cached = self._domain_age_cache_get(cache_key)
        if cached is not None:
            return cached

        score = self._domain_age_from_whois(domain)
        self._domain_age_cache_set(cache_key, score)
        return score

    def _domain_age_from_whois(self, domain: str) -> float:
        if not coerce_bool(self._cfg("domain_age.enable_whois"), False):
            return self._domain_age_fallback(domain)

        try:
            whois_module = __import__("whois")
            record = whois_module.whois(domain)
            creation_date = getattr(record, "creation_date", None) or (record.get("creation_date") if isinstance(record, Mapping) else None)
            if isinstance(creation_date, list):
                creation_date = creation_date[0] if creation_date else None
            if not creation_date:
                return clamp_score(self._cfg("domain_age.unknown_score"))
            if isinstance(creation_date, str):
                creation_date = parse_iso_datetime(creation_date)
            if creation_date.tzinfo is None:
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - creation_date.astimezone(timezone.utc)).days
            if age_days < coerce_int(self._cfg("domain_age.new_domain_days"), 30):
                return clamp_score(self._cfg("domain_age.new_domain_score"))
            if age_days < coerce_int(self._cfg("domain_age.young_domain_days"), 365):
                return clamp_score(self._cfg("domain_age.young_domain_score"))
            return clamp_score(self._cfg("domain_age.established_domain_score"))
        except Exception as exc:
            logger.info("WHOIS unavailable or failed; using fallback: %s", safe_log_payload(
                "domain_age_whois_fallback",
                {"domain": domain, "error_type": type(exc).__name__},
            ))
            return self._domain_age_fallback(domain)

    def _domain_age_fallback(self, domain: str) -> float:
        tld = domain.rsplit(".", 1)[-1].lower() if domain else ""
        tld_scores = self._cfg("domain_age.tld_scores", {})
        if isinstance(tld_scores, Mapping) and tld in tld_scores:
            return clamp_score(tld_scores[tld])
        established = {str(item).lower() for item in self._cfg("domain_age.established_tlds", [])}
        suspicious = {str(item).lower() for item in self._cfg("domain_age.suspicious_tlds", [])}
        if tld in suspicious:
            return clamp_score(self._cfg("domain_age.suspicious_tld_score"))
        if tld in established:
            return clamp_score(self._cfg("domain_age.established_tld_score"))
        return clamp_score(self._cfg("domain_age.unknown_score"))

    def _domain_age_cache_get(self, key: str) -> Optional[float]:
        if not coerce_bool(self._cfg("domain_age.cache_enabled"), True):
            return None
        entries = self.memory.recall(tag=key, top_k=1)
        if not entries:
            return None
        try:
            return clamp_score(entries[0]["data"]["score"])
        except Exception:
            return None

    def _domain_age_cache_set(self, key: str, score: float) -> None:
        if not coerce_bool(self._cfg("domain_age.cache_enabled"), True):
            return
        self._add_memory_entry(
            {"score": clamp_score(score), "timestamp": utc_iso()},
            tags=[key, "adaptive_security", "domain_age"],
            sensitivity=coerce_float(self._cfg("memory.domain_age_sensitivity"), 0.3, minimum=0.0, maximum=1.0),
            ttl_seconds=coerce_int(self._cfg("domain_age.cache_ttl_seconds"), 86400, minimum=60),
            purpose="adaptive_security_domain_age_cache",
            source="adaptive_security",
        )

    def _contains_ip_address(
        self,
        text: str,
        *,
        validate: bool = True,
        check_urls: bool = True,
        allow_private: bool = False,
        allow_reserved: bool = False,
    ) -> Union[bool, Tuple[bool, Optional[str]]]:
        """Enhanced IP-address detection with security-aware validation."""

        printer.status("ADAPT", "Detecting IP address", "info")
        if not text:
            return False if not validate else (False, "Empty input")

        candidate_text = str(text)
        if check_urls:
            for url_match in self._IP_IN_URL.finditer(candidate_text):
                host = url_match.group(1).strip("[]")
                if self._is_ip_like(host):
                    candidate_text += f" {host}"

        for match in self._IPV4_PATTERN.findall(candidate_text):
            ip_str = ".".join(match) if isinstance(match, tuple) else str(match)
            if validate:
                valid, reason = self._validate_ip(ip_str, 4, allow_private, allow_reserved)
                if valid:
                    return True, reason
            else:
                return True

        for raw_match in self._IPV6_PATTERN.findall(candidate_text):
            ip_str = self._clean_ipv6(str(raw_match))
            if validate:
                valid, reason = self._validate_ip(ip_str, 6, allow_private, allow_reserved)
                if valid:
                    return True, reason
            else:
                return True

        return False if not validate else (False, "No valid IP found")

    def _is_ip_like(self, host: str) -> bool:
        return bool(host) and ((":" in host) or ("." in host and any(char.isdigit() for char in host)))

    def _clean_ipv6(self, ip_str: str) -> str:
        cleaned = str(ip_str).strip("[]")
        if self._IP_WITH_PORT.search(cleaned) and cleaned.count(":") > 1:
            maybe_addr, maybe_port = cleaned.rsplit(":", 1)
            if maybe_port.isdigit():
                cleaned = maybe_addr
        return cleaned

    def _validate_ip(self, ip_str: str, version: int, allow_private: bool, allow_reserved: bool) -> Tuple[bool, str]:
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.version != version:
                return False, f"IP version mismatch ({ip.version} vs {version})"
            if ip.is_private and not allow_private:
                return True, "Private IP found"
            if ip.is_reserved and not allow_reserved:
                return True, "Reserved IP found"
            if ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return True, "Special-use IP found"
            if ip.is_global:
                return True, "Public routable IP found"
            return False, "Non-routable IP"
        except ValueError:
            return False, "Invalid IP format"

    def _extract_domain(self, url: str) -> str:
        try:
            return extract_domain(url)
        except SecurityError:
            parsed = urlparse(normalize_url(url))
            return (parsed.hostname or "").lower()
        except Exception as exc:
            raise wrap_security_exception(
                exc,
                operation="extract_domain",
                component="adaptive_security",
                context={"url_fingerprint": fingerprint(url)},
                error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                severity=SecuritySeverity.MEDIUM,
            ) from exc

    def monitor_traffic(self, packet: Mapping[str, Any]) -> Dict[str, Any]:
        """Analyze network traffic patterns and block suspicious sources in-memory."""

        printer.status("ADAPT", "Analyzing network traffic patterns", "info")
        if not isinstance(packet, Mapping):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Traffic monitor requires a mapping packet.",
                severity=SecuritySeverity.HIGH,
                context={"payload_type": type(packet).__name__},
                component="adaptive_security",
            )

        source_ip = str(packet.get("source_ip", "unknown"))
        anomalous = self._detect_anomalous_payload(packet)
        if anomalous:
            self._block_ip(source_ip, reason="anomalous_payload")
        return sanitize_for_logging({
            "source_ip": source_ip,
            "anomalous": anomalous,
            "blocked": source_ip in self.blocked_ips,
            "timestamp": utc_iso(),
        })

    def _detect_anomalous_payload(self, packet: Mapping[str, Any]) -> bool:
        payload = str(packet.get("payload", ""))
        payload_size = len(payload.encode("utf-8", errors="replace"))
        size_limit = coerce_int(self._cfg("traffic.payload_size_limit"), self.input_size_limit, minimum=1)
        return payload_size > size_limit or self._contains_binary_patterns(payload)

    def _contains_binary_patterns(self, data: str) -> bool:
        return bool(re.search(r"[\x00-\x08\x0e-\x1f]", data or ""))

    def _get_client_ip(self, request_obj: Optional[Any] = None) -> str:
        """Get client IP from explicit context, request-like objects, or safe fallback."""

        candidate = None
        if request_obj is not None:
            headers = getattr(request_obj, "headers", {}) or {}
            for header in self._cfg("client_ip_headers", []):
                raw = headers.get(header) if hasattr(headers, "get") else None
                if raw:
                    candidate = str(raw).split(",", 1)[0].strip()
                    break
            candidate = candidate or getattr(request_obj, "remote_addr", None)
        if not candidate:
            candidate = str(self._cfg("default_client_ip"))
        if not candidate:
            candidate = "127.0.0.1"
        return normalize_text(candidate, max_length=128)

    def _block_ip(self, ip: str, *, reason: str = "policy") -> None:
        normalized_ip = normalize_text(ip, max_length=128)
        self.blocked_ips[normalized_ip] = {"reason": reason, "timestamp": utc_iso(), "fingerprint": fingerprint(normalized_ip)}
        logger.info("IP blocked in adaptive-security memory: %s", safe_log_payload("adaptive_ip_blocked", {"ip": normalized_ip, "reason": reason}))

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit_decision(self, client_ip: Optional[str] = None) -> RateLimitDecision:
        principal = normalize_identifier(client_ip or self._get_client_ip(), max_length=128, default="unknown")
        now = time.time()
        window = coerce_int(self._cfg("rate_limit_window_seconds"), 60, minimum=1)
        limit = coerce_int(self._cfg("rate_limit"), 30, minimum=1)

        queue = self.request_tracker[principal]
        while queue and now - queue[0] > window:
            queue.popleft()
        queue.append(now)

        if len(queue) > limit:
            retry_after = max(0.0, window - (now - queue[0]))
            logger.warning("AdaptiveSecurity rate limit exceeded: %s", safe_log_payload("adaptive_rate_limit_exceeded", {"principal": principal, "count": len(queue), "limit": limit}))
            return RateLimitDecision(False, principal, len(queue), limit, window, retry_after, "rate_limit_exceeded")
        return RateLimitDecision(True, principal, len(queue), limit, window)

    def _check_input_overload(self) -> bool:
        """Legacy boolean overload check retained for subsystem compatibility."""

        return not self._rate_limit_decision(self._get_client_ip()).allowed

    # ------------------------------------------------------------------
    # Supply-chain checks
    # ------------------------------------------------------------------

    def _load_trusted_hashes(self) -> Dict[str, str]:
        configured_hashes = self._cfg("trusted_hashes", {})
        if isinstance(configured_hashes, Mapping):
            return {str(name): str(value).lower() for name, value in configured_hashes.items() if value}

        if isinstance(configured_hashes, str) and configured_hashes:
            try:
                loaded = json.loads(load_text_file(configured_hashes, max_bytes=coerce_int(self._cfg("supply_chain.max_hash_file_bytes"), 262144)))
                if isinstance(loaded, Mapping):
                    return {str(name): str(value).lower() for name, value in loaded.items() if value}
            except Exception as exc:
                raise wrap_security_exception(
                    exc,
                    operation="load_trusted_hashes",
                    component="adaptive_security",
                    context={"trusted_hashes_path": configured_hashes},
                    error_type=SecurityErrorType.CONFIGURATION_TAMPERING,
                    severity=SecuritySeverity.HIGH,
                ) from exc
        return {}

    def check_supply_chain(self, file_path: str) -> Dict[str, Any]:
        """Check a file against configured trusted hashes."""

        printer.status("ADAPT", "Checking supply chain", "info")
        path = Path(file_path)
        max_bytes = coerce_int(self._cfg("supply_chain.max_file_bytes"), 10485760, minimum=1024)
        block_unknown = coerce_bool(self._cfg("supply_chain.block_unknown_hash"), True)
        allowed_extensions = {str(ext).lower() for ext in self._cfg("supply_chain.allowed_extensions", [])}

        if not path.exists() or not path.is_file():
            raise UnauthorizedAccessError(resource=str(path), policy_violated="supply_chain_file_must_exist", attempted_action="hash_check", component="adaptive_security")

        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            raise SupplyChainCompromiseError(dependency=path.name, indicator="disallowed_file_extension", version=path.suffix, component="adaptive_security")

        size = path.stat().st_size
        if size > max_bytes:
            raise ResourceExhaustionError(resource_type="supply_chain_file_bytes", current_usage=float(size), limit=float(max_bytes), source_identifier=str(path), component="adaptive_security")

        file_hash = self._calculate_file_hash(str(path))
        trusted_by_name = {name: digest for name, digest in self.safe_package_hashes.items() if digest == file_hash}
        is_trusted = bool(trusted_by_name)
        risk_level = "low" if is_trusted else ("high" if block_unknown else "medium")
        decision = "allow" if is_trusted else ("block" if block_unknown else "review")
        result = SupplyChainCheckResult(
            file_fingerprint=fingerprint(str(path)),
            file_hash=file_hash,
            is_trusted=is_trusted,
            risk_level=risk_level,
            decision=decision,
            matched_name=next(iter(trusted_by_name.keys()), None),
            metadata={"size_bytes": size, "extension": path.suffix.lower()},
            timestamp=utc_iso(),
        )
        if not is_trusted and block_unknown:
            logger.warning("Untrusted supply-chain artifact: %s", safe_log_payload("untrusted_supply_chain_artifact", result.to_dict()))
        return result.to_dict()

    def _calculate_file_hash(self, file_path: str) -> str:
        algorithm = safe_hash_algorithm(str(self._cfg("supply_chain.hash_algorithm")))
        hasher = hashlib.new(algorithm)
        chunk_size = coerce_int(self._cfg("supply_chain.chunk_size"), 65536, minimum=4096)
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_phishing_model(
        self,
        model_type: str,
        training_data: List[Tuple[List[float], List[float]]],
        *,
        validation_data: Optional[List[Tuple[List[float], List[float]]]] = None,
    ) -> Dict[str, Any]:
        """Retrain a specific neural network with new security data."""

        selected = self._select_model(model_type)
        target_nn = selected["model"]
        model_save_path = selected["path"]
        training_cfg = self._cfg("training", {})

        if not training_data:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Training data must not be empty.", severity=SecuritySeverity.HIGH, context={"model_type": model_type}, component="adaptive_security")

        max_samples = coerce_int(training_cfg.get("max_samples"), 10000, minimum=1)
        if len(training_data) > max_samples:
            raise ResourceExhaustionError(resource_type="adaptive_training_samples", current_usage=float(len(training_data)), limit=float(max_samples), source_identifier=model_type, component="adaptive_security")

        target_nn.train(
            training_data,
            epochs=coerce_int(training_cfg.get("epochs"), 50, minimum=1),
            initial_learning_rate=coerce_float(training_cfg.get("learning_rate"), 0.001, minimum=0.0),
            batch_size=coerce_int(training_cfg.get("batch_size"), 32, minimum=1),
            validation_data=validation_data or [],
            verbose=coerce_bool(training_cfg.get("verbose"), False),
            print_every_n_epochs=coerce_int(training_cfg.get("print_every_n_epochs"), 10, minimum=1),
            save_best_model_path=model_save_path if coerce_bool(training_cfg.get("save_best_model"), True) else None,
        )
        if coerce_bool(training_cfg.get("save_final_model"), True):
            target_nn.save_model(model_save_path)

        summary = {"model_type": model_type, "model_path_fingerprint": fingerprint(model_save_path), "samples": len(training_data), "saved": True, "timestamp": utc_iso()}
        logger.info("Adaptive model training completed: %s", safe_log_payload("adaptive_model_training_completed", summary))
        return sanitize_for_logging(summary)

    def _select_model(self, model_type: str) -> Dict[str, Any]:
        normalized = normalize_identifier(model_type, default="unknown")
        if normalized == "email":
            return {"model": self.email_nn, "path": self.email_model_path}
        if normalized == "url":
            return {"model": self.url_nn, "path": self.url_model_path}
        raise ConfigurationTamperingError("adaptive_security.model_type", f"Unknown model type for training: {model_type}", component="adaptive_security")

    # ------------------------------------------------------------------
    # Audit and memory
    # ------------------------------------------------------------------

    def _store_analysis_result(self, result: AdaptiveAnalysisResult) -> None:
        if not coerce_bool(self._cfg("memory.store_analysis"), True):
            return
        try:
            self._add_memory_entry(
                result.to_dict(),
                tags=list(self._cfg("memory.analysis_tags", ["adaptive_security", "analysis", result.source_type])),
                sensitivity=coerce_float(self._cfg("memory.analysis_sensitivity"), 0.7, minimum=0.0, maximum=1.0),
                ttl_seconds=coerce_int(self._cfg("memory.analysis_ttl_seconds"), 86400, minimum=60),
                purpose="adaptive_security_analysis",
                source="adaptive_security",
                metadata={"source_type": result.source_type, "decision": result.decision, "risk_level": result.risk_level},
            )
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_store_error"), False):
                raise AuditLogFailureError(logging_target="secure_memory.adaptive_analysis", failure_mode=type(exc).__name__, component="adaptive_security", cause=exc) from exc
            logger.warning("Adaptive analysis memory store failed: %s", safe_log_payload("adaptive_memory_store_failed", {"error_type": type(exc).__name__}))


if __name__ == "__main__":
    print("\n=== Running Adaptive Security ===\n")
    printer.status("TEST", "Adaptive Security initialized", "info")

    security_system = AdaptiveSecurity()
    security_system._get_client_ip = lambda request_obj=None: "198.51.100.10"

    assert security_system._contains_ip_address("Server: 192.168.1.1", validate=False) is True

    valid, reason = security_system._contains_ip_address("Visit https://[2001:db8::1]/admin", validate=True, check_urls=True) # pyright: ignore[reportGeneralTypeIssues]
    assert valid is True
    assert isinstance(reason, str)

    valid_private, private_reason = security_system._contains_ip_address("Internal service at 10.0.0.5:3000", validate=True, allow_private=False) # pyright: ignore[reportGeneralTypeIssues]
    assert valid_private is True
    assert "Private" in private_reason or "Special" in private_reason # pyright: ignore[reportOperatorIssue]

    sample_email = {
        "from": "Fake Bank <support@fakebank-login.example>",
        "reply_to": "security@fakebank-login.example",
        "subject": "Urgent: Your Account Needs Verification",
        "body": "Verify now at http://198.51.100.25/login?token=secret or your account will be suspended.",
        "links": ["http://198.51.100.25/login?token=secret"],
        "attachments": [{"filename": "invoice.exe", "size": 2048}],
        "timestamp": time.time(),
    }
    email_analysis = security_system.analyze_email(sample_email, client_ip="198.51.100.10")
    assert "phishing_score" in email_analysis
    assert len(email_analysis["features"]) == int(security_system._cfg("feature_schema.email_num_inputs"))
    assert "secret" not in stable_json(email_analysis)
    printer.pretty("Email Analysis", email_analysis, "success")

    url_analysis = security_system.analyze_url("http://198.51.100.25/login?token=secret&redirect=http://evil.example", client_ip="198.51.100.11")
    assert "phishing_score" in url_analysis
    assert len(url_analysis["features"]) == int(security_system._cfg("feature_schema.url_num_inputs"))
    assert "secret" not in stable_json(url_analysis)
    printer.pretty("URL Analysis", url_analysis, "success")

    packet_result = security_system.monitor_traffic({"source_ip": "203.0.113.5", "payload": "GET /login\x00\x01"})
    assert packet_result["anomalous"] is True

    test_artifact = Path("adaptive_security_self_test_artifact.bin")
    test_artifact.write_bytes(b"known-good-adaptive-security-artifact")
    digest = security_system._calculate_file_hash(str(test_artifact))
    security_system.safe_package_hashes["adaptive_security_self_test_artifact.bin"] = digest
    supply_chain = security_system.check_supply_chain(str(test_artifact))
    assert supply_chain["is_trusted"] is True
    test_artifact.unlink(missing_ok=True)
    printer.pretty("Supply Chain Check", supply_chain, "success")

    print("\n=== Test ran successfully ===\n")
