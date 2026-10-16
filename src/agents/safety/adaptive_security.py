"""Adaptive phishing/security analysis for the SLAI Safety Agent subsystem."""
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
from threading import RLock
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from .modules.neural_network import *
from .modules.safety_features import SafetyFeatures
from .secure_memory import SecureMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Adaptive Security System")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
ANALYSIS_SCHEMA_VERSION = "adaptive_security.analysis.v4"
SUPPLY_CHAIN_SCHEMA_VERSION = "adaptive_security.supply_chain.v3"


@dataclass(frozen=True)
class AdaptiveAnalysisResult:
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
        data["module_version"] = MODULE_VERSION
        data["timestamp"] = self.timestamp or utc_iso()
        data["features"] = [float(v) for v in data.get("features", [])]
        data["metadata"] = sanitize_for_logging(data.get("metadata", {}))
        return data


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    principal: str
    count: int
    limit: int
    window_seconds: int
    retry_after_seconds: float = 0.0
    reason: str = "ok"
    applied: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging(asdict(self))


@dataclass(frozen=True)
class SupplyChainCheckResult:
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
        data["module_version"] = MODULE_VERSION
        data["timestamp"] = self.timestamp or utc_iso()
        data["metadata"] = sanitize_for_logging(data.get("metadata", {}))
        return data


class AdaptiveSecurity:
    """Phishing-focused adaptive security with explainable deterministic fallback."""

    _IPV4_PATTERN = re.compile(r"\b(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3}\b")
    _IPV6_PATTERN = re.compile(r"\b(?:[A-F0-9]{1,4}:){2,7}[A-F0-9]{0,4}\b", re.IGNORECASE)
    _IP_IN_URL = re.compile(r"(?i)\bhttps?://([^/\s?#]+)")

    def __init__(
        self,
        *,
        memory: Optional[SecureMemory] = None,
        safety_features: Optional[SafetyFeatures] = None,
    ) -> None:
        self.config = load_global_config()
        self.adaptive_config = get_config_section("adaptive_security")
        self._validate_configuration()
        self.memory = memory or SecureMemory.shared()
        self.safety_features = safety_features or SafetyFeatures()
        self.rate_limit = coerce_int(self._cfg("rate_limit", 30), 30, minimum=1)
        self.input_size_limit = coerce_int(self._cfg("input_size_limit", 65536), 65536, minimum=1)
        self.phishing_threshold = clamp_score(self._cfg("phishing_threshold", 0.70))
        self.review_threshold = clamp_score(self._cfg("review_threshold", 0.45))
        self.block_threshold = clamp_score(self._cfg("block_threshold", 0.80))
        self.email_model_path = str(self._cfg("email_model_path", ""))
        self.url_model_path = str(self._cfg("url_model_path", ""))
        maxlen = coerce_int(self._cfg("rate_limit_tracker_maxlen", 200), 200, minimum=1)
        self.request_tracker: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        self.blocked_ips: Dict[str, Dict[str, Any]] = {}
        self._state_lock = RLock()
        self.safe_package_hashes = self._load_trusted_hashes()
        self.email_nn = self._initialize_neural_network("email")
        self.url_nn = self._initialize_neural_network("url")

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.adaptive_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.adaptive_config, Mapping):
            raise ConfigurationTamperingError("adaptive_security", "adaptive_security config must be a mapping", component="adaptive_security")
        if clamp_score(self.adaptive_config.get("review_threshold", 0.45)) > clamp_score(self.adaptive_config.get("block_threshold", 0.80)):
            raise ConfigurationTamperingError("adaptive_security.review_threshold", "review_threshold must be <= block_threshold", component="adaptive_security")

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _model_settings(self, model_type: str) -> Dict[str, Any]:
        if model_type not in {"email", "url"}:
            raise ConfigurationTamperingError("adaptive_security.model_type", f"Unsupported model type: {model_type}", component="adaptive_security")
        schema = self._cfg("feature_schema", {}) or {}
        return {
            "model_type": model_type,
            "num_inputs": coerce_int(schema.get(f"{model_type}_num_inputs", 11 if model_type == "email" else 8), 11 if model_type == "email" else 8, minimum=1),
            "model_path": str(self._cfg(f"{model_type}_model_path", "")),
            "layer_config": list(self._cfg(f"{model_type}_layer_config", [])),
            "loss_function_name": str(self._cfg("model_defaults.loss_function_name", "cross_entropy")),
            "optimizer_name": str(self._cfg("model_defaults.optimizer_name", "adam")),
            "problem_type": str(self._cfg("model_defaults.problem_type", "binary_classification")),
        }

    def _initialize_neural_network(self, model_type: str) -> Optional[NeuralNetwork]:
        """Load a verified persisted model or enter deterministic degraded mode.

        Compatibility failures are handled separately from integrity failures:
        - a legacy/unsupported persistence schema never gets loaded implicitly;
        - malformed/tampered current-schema artifacts are still treated as
          integrity failures and can fail closed through configuration;
        - an unavailable model contributes no model score.
        """
        settings = self._model_settings(model_type)
        path = settings["model_path"]
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()

        if path and resolved.is_file():
            artifact_schema = self._read_model_artifact_schema(resolved)
            if artifact_schema != MODEL_SCHEMA_VERSION:
                return self._handle_incompatible_model_artifact(
                    model_type=model_type,
                    path=resolved,
                    observed_schema=artifact_schema,
                )

            try:
                model = NeuralNetwork.load_model(
                    str(resolved),
                    custom_config_override=get_config_section("neural_network"),
                )
                if getattr(model, "num_inputs", settings["num_inputs"]) != settings["num_inputs"]:
                    raise SecurityError(
                        SecurityErrorType.MODEL_TAMPERING,
                        "Adaptive-security model input schema mismatch.",
                        component="adaptive_security",
                        context={
                            "model_type": model_type,
                            "expected_inputs": settings["num_inputs"],
                            "observed_inputs": getattr(model, "num_inputs", None),
                            "model_path_fingerprint": fingerprint(str(resolved)),
                        },
                    )

                try:
                    setattr(model, "_adaptive_verified_loaded_model", True)
                except Exception:
                    pass

                self._audit_model_load(model_type, path, model)
                return model

            except SecurityError as exc:
                if coerce_bool(self._cfg("model_loading.fail_closed_on_load_error", False), False):
                    raise

                logger.warning(
                    "Adaptive %s model failed integrity/persistence validation; deterministic degraded mode is active: %s",
                    model_type,
                    safe_log_payload(
                        "adaptive_model_load_security_failure",
                        {
                            "model_type": model_type,
                            "model_path_fingerprint": fingerprint(str(resolved)),
                            "error_type": type(exc).__name__,
                        },
                    ),
                )
                return None

            except Exception as exc:
                if coerce_bool(self._cfg("model_loading.fail_closed_on_load_error", False), False):
                    raise wrap_security_exception(
                        exc,
                        operation=f"load_{model_type}_model",
                        component="adaptive_security",
                        error_type=SecurityErrorType.MODEL_TAMPERING,
                        severity=SecuritySeverity.HIGH,
                    ) from exc

                logger.warning(
                    "Adaptive %s model unavailable; deterministic degraded mode is active: %s",
                    model_type,
                    safe_log_payload(
                        "adaptive_model_load_failure",
                        {
                            "model_type": model_type,
                            "model_path_fingerprint": fingerprint(str(resolved)),
                            "error_type": type(exc).__name__,
                        },
                    ),
                )
                return None

        # An untrained network is never used as production evidence.
        environment = normalize_identifier(
            get_nested(get_config_section("neural_network"), "environment", "production"),
            max_length=32,
        )
        allow_dev = (
            coerce_bool(self._cfg("model_loading.allow_untrained_fallback", False), False)
            and coerce_bool(self._cfg("model_loading.permit_untrained_development_model", False), False))
        if (
            allow_dev
            and environment not in {"production", "prod"}
            and settings["layer_config"]
        ):
            logger.warning("Using explicitly enabled untrained development model for %s; score will be marked non-evidentiary.", model_type)
            return NeuralNetwork(
                num_inputs=settings["num_inputs"],
                layer_config=settings["layer_config"],
                loss_function_name=settings["loss_function_name"],
                optimizer_name=settings["optimizer_name"],
                problem_type=settings["problem_type"],
                config=get_config_section("neural_network"),
            )

        return None

    def _read_model_artifact_schema(self, path: Path) -> Optional[str]:
        """Read only the persistence schema from a bounded JSON artifact."""
        max_bytes = coerce_int(
            self._cfg("model_loading.max_model_bytes", 50_000_000),
            50_000_000,
            minimum=4096,
        )

        try:
            size = path.stat().st_size
        except OSError as exc:
            raise wrap_security_exception(
                exc,
                operation="inspect_model_artifact",
                component="adaptive_security",
                context={"model_path_fingerprint": fingerprint(str(path))},
                error_type=SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                severity=SecuritySeverity.HIGH,
            ) from exc

        if size > max_bytes:
            raise ResourceExhaustionError(
                "adaptive_model_file_bytes",
                float(size),
                float(max_bytes),
                source_identifier="adaptive_security",
                component="adaptive_security",
            )

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if coerce_bool(
                self._cfg("model_loading.fail_closed_on_load_error", False),
                False,
            ):
                raise wrap_security_exception(
                    exc,
                    operation="inspect_model_artifact",
                    component="adaptive_security",
                    context={"model_path_fingerprint": fingerprint(str(path))},
                    error_type=SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                    severity=SecuritySeverity.HIGH,
                ) from exc

            logger.warning(
                "Adaptive model artifact could not be inspected; deterministic "
                "degraded mode is active: %s",
                safe_log_payload(
                    "adaptive_model_artifact_unreadable",
                    {
                        "model_path_fingerprint": fingerprint(str(path)),
                        "error_type": type(exc).__name__,
                    },
                ),
            )
            return None

        if not isinstance(payload, Mapping):
            if coerce_bool(
                self._cfg("model_loading.fail_closed_on_load_error", False),
                False,
            ):
                raise SecurityError(
                    SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                    "Adaptive-security model artifact root must be a mapping.",
                    component="adaptive_security",
                    context={"model_path_fingerprint": fingerprint(str(path))},
                )
            return None

        schema = payload.get("schema_version")
        return str(schema) if schema not in (None, "") else None

    def _handle_incompatible_model_artifact(
        self,
        *,
        model_type: str,
        path: Path,
        observed_schema: Optional[str],
    ) -> Optional[NeuralNetwork]:
        """Reject legacy persistence schemas without treating them as inference data."""
        fail_closed = coerce_bool(
            self._cfg("model_loading.fail_closed_on_schema_mismatch", False),
            False,
        )
        context = {
            "model_type": model_type,
            "expected_schema": MODEL_SCHEMA_VERSION,
            "observed_schema": observed_schema or "missing",
            "model_path_fingerprint": fingerprint(str(path)),
        }

        if fail_closed:
            raise SecurityError(
                SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                "Adaptive-security model persistence schema is incompatible.",
                component="adaptive_security",
                context=context,
            )

        logger.warning(
            "Adaptive %s model uses an incompatible persistence schema "
            "(observed=%s, expected=%s); deterministic degraded mode is active. "
            "Retrain and re-save this model with the current NeuralNetwork.",
            model_type,
            observed_schema or "missing",
            MODEL_SCHEMA_VERSION,
        )

        try:
            self.memory.add(
                {
                    "event": "adaptive_model_schema_incompatible",
                    **context,
                    "timestamp": utc_iso(),
                },
                tags=["adaptive_security", "model", "compatibility"],
                sensitivity=coerce_float(
                    self._cfg("memory.model_event_sensitivity", 0.65),
                    0.65,
                    minimum=0.0,
                    maximum=1.0,
                ),
                purpose="adaptive_security_model_governance",
                owner="adaptive_security",
                source="adaptive_security",
                metadata={
                    "eligible_for_compliance": True,
                    "requires_model_refresh": True,
                },
            )
        except Exception as exc:
            logger.warning(
                "Could not store adaptive model compatibility event: %s",
                type(exc).__name__,
            )

        return None

    def _audit_model_load(self, model_type: str, model_path: str, model: NeuralNetwork) -> None:
        if not coerce_bool(self._cfg("memory.store_model_events", True), True):
            return
        self.memory.add(
            {"event": "adaptive_model_loaded", "model_type": model_type, "model_path_fingerprint": fingerprint(model_path), "num_inputs": getattr(model, "num_inputs", None), "timestamp": utc_iso()},
            tags=list(self._cfg("memory.model_event_tags", ["adaptive_security", "model"])), sensitivity=coerce_float(self._cfg("memory.model_event_sensitivity", 0.65), 0.65),
            purpose="adaptive_security_model_governance", owner="adaptive_security", source="adaptive_security", metadata={"eligible_for_compliance": True},
        )

    # ------------------------------------------------------------------
    # Public analysis
    # ------------------------------------------------------------------
    def analyze_email(self, email: Mapping[str, Any], *, client_ip: Optional[str] = None, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if not isinstance(email, Mapping):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Email analysis requires a mapping.", component="adaptive_security")
        self._validate_payload_size(email, source_type="email")
        feature_result = self.safety_features.assess_email_risk(email) if hasattr(self.safety_features, "assess_email_risk") else None
        features = self._extract_email_features(email)
        local_text = normalize_text(f"{email.get('from','')} {email.get('subject','')} {email.get('body','')}", max_length=coerce_int(self._cfg("max_text_length", 4096), 4096), lowercase=True)
        return self._analyze_features(features, "email", self.email_nn, client_ip=client_ip, context=context, feature_result=feature_result, local_text=local_text)

    def analyze_url(self, url: str, *, client_ip: Optional[str] = None, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        normalized_url = normalize_url(url)
        self._validate_payload_size(normalized_url, source_type="url")
        feature_result = self.safety_features.assess_url_risk(normalized_url) if hasattr(self.safety_features, "assess_url_risk") else None
        features = self._extract_url_features(normalized_url)
        return self._analyze_features(features, "url", self.url_nn, client_ip=client_ip, context=context, feature_result=feature_result, local_text=normalized_url.lower())

    def _analyze_features(
        self,
        features: Sequence[float],
        source_type: str,
        model: Optional[NeuralNetwork],
        *,
        client_ip: Optional[str],
        context: Optional[Mapping[str, Any]],
        feature_result: Optional[Any],
        local_text: str,
    ) -> Dict[str, Any]:
        principal = self._principal_from_context(client_ip, context)
        rate_decision = self._rate_limit_decision(principal)
        if not rate_decision.allowed:
            return self._blocked_rate_limit_result(source_type, features, rate_decision)

        heuristic_score, indicators = self._heuristic_score(source_type, features, feature_result)
        model_score: Optional[float] = None
        model_evidentiary = False
        if model is not None:
            try:
                prediction = model.predict(list(features))
                if isinstance(prediction, Sequence) and not isinstance(prediction, (str, bytes)):
                    model_score = clamp_score(prediction[0] if prediction else 0.0)
                else:
                    model_score = clamp_score(prediction)
                model_evidentiary = self._model_is_evidentiary(model)
            except Exception as exc:
                if coerce_bool(self._cfg("model_loading.fail_closed_on_prediction_error", False), False):
                    raise wrap_security_exception(exc, operation=f"predict_{source_type}", component="adaptive_security", error_type=SecurityErrorType.UNSAFE_MODEL_STATE, severity=SecuritySeverity.HIGH) from exc
                logger.warning("Adaptive %s model prediction failed; using deterministic score only.", source_type)
                model_score = None

        weights = self._cfg("score_weights", {}) or {}
        if model_score is not None and model_evidentiary:
            score = weighted_average({"model": model_score, "heuristic": heuristic_score}, weights, default=max(model_score, heuristic_score))
            confidence = self._confidence_score(model_score, heuristic_score)
        else:
            score = heuristic_score
            confidence = clamp_score(coerce_float(self._cfg("heuristic_only_confidence", 0.78), 0.78))
        score = clamp_score(score)
        decision = threshold_decision(score, block_threshold=self.block_threshold, review_threshold=self.review_threshold)
        threat_type = self._determine_threat_type(features, source_type, indicators=indicators, local_text=local_text)
        result = AdaptiveAnalysisResult(
            source_type=source_type,
            phishing_score=score,
            is_phishing=score >= self.phishing_threshold,
            decision=decision,
            risk_level=categorize_risk(score),
            threat_type=threat_type,
            features=[float(v) for v in features],
            model_score=float(model_score) if model_score is not None else 0.0,
            heuristic_score=heuristic_score,
            confidence=confidence,
            indicators=dedupe_preserve_order(indicators),
            metadata={
                "model_available": model is not None,
                "model_evidentiary": model_evidentiary,
                "degraded_mode": not model_evidentiary,
                "principal_fingerprint": fingerprint(principal) if principal else None,
                "rate_limit": rate_decision.to_dict(),
                "feature_schema": source_type,
            },
        )
        self._store_analysis_result(result)
        return result.to_dict()

    def _model_is_evidentiary(self, model: NeuralNetwork) -> bool:
        summary = getattr(model, "last_training_summary", None)
        if summary:
            return True
        # Loaded signed model artifacts are considered evidentiary even when the
        # implementation does not expose a training summary in memory.
        return bool(
            getattr(model, "_adaptive_verified_loaded_model", False)
            or getattr(model, "_loaded_from_signed_artifact", False)
            or getattr(model, "loaded_from_persistence", False)
            or getattr(model, "model_signature", None)
        )

    def _blocked_rate_limit_result(self, source_type: str, features: Sequence[float], decision: RateLimitDecision) -> Dict[str, Any]:
        return AdaptiveAnalysisResult(source_type, 1.0, True, "block", "critical", "Rate-limit / abuse-control trigger", list(features), 0.0, 1.0, 0.95, ["rate_limit_exceeded"], {"rate_limit": decision.to_dict(), "model_evidentiary": False}).to_dict()

    # ------------------------------------------------------------------
    # Features and heuristics
    # ------------------------------------------------------------------
    def _extract_email_features(self, email: Mapping[str, Any]) -> List[float]:
        if hasattr(self.safety_features, "extract_email_feature_vector"):
            return self._validate_features(self.safety_features.extract_email_feature_vector(email), "email")
        legacy_features: Any = self.safety_features
        links = email.get("links", []) if isinstance(email.get("links", []), list) else []
        features = [
            len(str(email.get("from", ""))), len(str(email.get("subject", ""))),
            legacy_features._suspicious_keyword_score(str(email.get("subject", "")) + " " + str(email.get("body", ""))),
            len(links), legacy_features._urgent_language_score(email),
            1.0 if email.get("attachments") else 0.0, legacy_features._domain_mismatch_score(dict(email)),
            sum(len(str(v)) for v in links) / len(links) if links else 0.0,
            legacy_features._ssl_cert_score(links), legacy_features._unusual_sender_score(email.get("from", "")),
            legacy_features._unusual_time_score(email.get("timestamp", 0)),
        ]
        return self._validate_features(features, "email")

    def _extract_url_features(self, url: str) -> List[float]:
        domain_age = self._domain_age(url)
        if hasattr(self.safety_features, "extract_url_feature_vector"):
            try:
                features = self.safety_features.extract_url_feature_vector(url, domain_age_score=domain_age)
            except TypeError:
                features = self.safety_features.extract_url_feature_vector(url)
                if len(features) >= 7:
                    features[6] = domain_age
            return self._validate_features(features, "url")
        legacy_features: Any = self.safety_features
        features = [
            len(url), legacy_features._url_entropy(url), legacy_features._num_subdomains(url),
            1.0 if self._contains_ip_address(url, validate=False) else 0.0, 1.0 if url.lower().startswith("https://") else 0.0,
            legacy_features._redirect_count(url), domain_age, legacy_features._special_char_count(url),
        ]
        return self._validate_features(features, "url")

    def _validate_features(self, features: Sequence[Any], source_type: str) -> List[float]:
        expected = coerce_int(get_nested(self._cfg("feature_schema", {}), f"{source_type}_num_inputs", 11 if source_type == "email" else 8), 11 if source_type == "email" else 8, minimum=1)
        if len(features) != expected:
            raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Adaptive-security feature vector dimension mismatch.", component="adaptive_security", context={"source_type": source_type, "expected": expected, "observed": len(features)})
        clean: List[float] = []
        for index, value in enumerate(features):
            numeric = coerce_float(value, float("nan"))
            if not math.isfinite(numeric):
                raise SecurityError(SecurityErrorType.UNSAFE_MODEL_STATE, "Non-finite adaptive-security feature.", component="adaptive_security", context={"index": index, "source_type": source_type})
            clean.append(float(numeric))
        return clean

    def _heuristic_score(self, source_type: str, features: Sequence[float], feature_result: Optional[Any]) -> Tuple[float, List[str]]:
        if feature_result is not None:
            score = clamp_score(feature_result.get("risk_score", 0.0) if isinstance(feature_result, Mapping) else getattr(feature_result, "risk_score", 0.0))
            indicators = feature_result.get("indicators", []) if isinstance(feature_result, Mapping) else getattr(feature_result, "indicators", [])
            return score, dedupe_preserve_order(str(v) for v in (indicators or []))
        if source_type == "email":
            scores = {"keywords": clamp_score(features[2]), "urgent": clamp_score(features[4]), "attachment": clamp_score(features[5]), "mismatch": clamp_score(features[6]), "sender": clamp_score(features[9]), "time": clamp_score(features[10])}
        else:
            scores = {"entropy": clamp_score(features[1]), "subdomains": clamp_score(features[2] / 5.0), "ip": clamp_score(features[3]), "insecure": clamp_score(1.0 - features[4]), "redirect": clamp_score(features[5]), "domain_age": clamp_score(features[6])}
        weights = self._cfg(f"heuristic_weights.{source_type}", {}) or {}
        threshold = clamp_score(self._cfg("indicator_threshold", 0.65))
        indicators = [name for name, value in scores.items() if value >= threshold]
        return weighted_average(scores, weights, default=max(scores.values() or [0.0])), indicators

    def _confidence_score(self, model_score: float, heuristic_score: float) -> float:
        agreement = 1.0 - abs(clamp_score(model_score) - clamp_score(heuristic_score))
        strength = max(abs(model_score - 0.5), abs(heuristic_score - 0.5)) * 2.0
        return clamp_score(0.55 * agreement + 0.45 * strength)

    def _determine_threat_type(self, features: Sequence[float], source_type: str, *, indicators: Sequence[str], local_text: str) -> Optional[str]:
        if source_type == "email":
            brands = [normalize_text(v, lowercase=True) for v in self._cfg("financial_brand_indicators", [])]
            if features[4] >= 0.75 and any(v and v in local_text for v in brands):
                return "Financial scam / payment-service impersonation indicators"
            if features[4] >= 0.75 and features[2] >= 0.6:
                return "Urgent social-engineering indicators"
            if "attachment_risk" in indicators:
                return "Suspicious attachment indicators"
            if "sender_reputation" in indicators or "domain_mismatch" in indicators:
                return "Sender/domain reputation anomaly"
            if self._detect_advanced_threat_pattern(list(features), source_type):
                return "Advanced or coordinated threat heuristic (attribution not established)"
        if source_type == "url":
            if features[3] >= 0.9 and features[4] < 0.5: return "Direct-IP insecure URL indicators"
            if features[3] >= 0.9: return "Suspicious IP-literal URL"
            if features[5] >= 0.75: return "Multi-hop redirect indicators"
            if "credential_in_url" in indicators: return "URL-embedded credential exposure"
            if "punycode_or_homograph" in indicators: return "Homograph / punycode impersonation indicators"
            if self._detect_advanced_threat_pattern(list(features), source_type): return "Advanced or coordinated threat heuristic (attribution not established)"
        return None

    def _detect_advanced_threat_pattern(self, features: List[float], source_type: str) -> bool:
        if source_type == "email":
            score = 0.30 * clamp_score(features[6]) + 0.20 * clamp_score(features[9]) + 0.25 * clamp_score(features[4]) + 0.15 * clamp_score(features[2]) + 0.10 * clamp_score(features[10])
            return score > clamp_score(self._cfg("threat_thresholds.email.apt_score", 0.65)) and features[3] > 1
        return clamp_score(features[3]) > 0.70 and clamp_score(features[6]) > 0.60 and clamp_score(features[1]) > clamp_score(self._cfg("threat_thresholds.url.apt_entropy", 0.55)) and clamp_score(features[5]) > 0.20

    def _validate_payload_size(self, payload: Any, *, source_type: str) -> None:
        size = len(stable_json(payload).encode("utf-8", errors="replace"))
        if size > self.input_size_limit:
            raise ResourceExhaustionError(f"{source_type}_payload_bytes", float(size), float(self.input_size_limit), source_identifier="adaptive_security", component="adaptive_security")

    # ------------------------------------------------------------------
    # Domain intelligence
    # ------------------------------------------------------------------
    def _domain_age(self, url: str) -> float:
        domain = self._extract_domain(url)
        if not domain: return clamp_score(self._cfg("domain_age.unknown_score", 0.6))
        key = f"domain_age:{fingerprint(domain)}"
        cached = self._domain_age_cache_get(key)
        if cached is not None: return cached
        score = self._domain_age_from_whois(domain)
        self._domain_age_cache_set(key, score)
        return score

    def _domain_age_from_whois(self, domain: str) -> float:
        if not coerce_bool(self._cfg("domain_age.enable_whois", False), False):
            return self._domain_age_fallback(domain)
        try:
            whois_module = __import__("whois")
            record = whois_module.whois(domain)
            creation = getattr(record, "creation_date", None) or (record.get("creation_date") if isinstance(record, Mapping) else None)
            if isinstance(creation, list): creation = creation[0] if creation else None
            if not creation: return clamp_score(self._cfg("domain_age.unknown_score", 0.6))
            if isinstance(creation, str): creation = parse_iso_datetime(creation)
            if creation.tzinfo is None: creation = creation.replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - creation.astimezone(timezone.utc)).days
            if days < coerce_int(self._cfg("domain_age.new_domain_days", 30), 30): return clamp_score(self._cfg("domain_age.new_domain_score", 0.9))
            if days < coerce_int(self._cfg("domain_age.young_domain_days", 365), 365): return clamp_score(self._cfg("domain_age.young_domain_score", 0.65))
            return clamp_score(self._cfg("domain_age.established_domain_score", 0.2))
        except Exception:
            return self._domain_age_fallback(domain)

    def _domain_age_fallback(self, domain: str) -> float:
        tld = domain.rsplit(".", 1)[-1].lower() if domain else ""
        scores = self._cfg("domain_age.tld_scores", {}) or {}
        if tld in scores: return clamp_score(scores[tld])
        if tld in {str(v).lower() for v in self._cfg("domain_age.suspicious_tlds", [])}: return clamp_score(self._cfg("domain_age.suspicious_tld_score", 0.8))
        if tld in {str(v).lower() for v in self._cfg("domain_age.established_tlds", [])}: return clamp_score(self._cfg("domain_age.established_tld_score", 0.3))
        return clamp_score(self._cfg("domain_age.unknown_score", 0.6))

    def _domain_age_cache_get(self, key: str) -> Optional[float]:
        if not coerce_bool(self._cfg("domain_age.cache_enabled", True), True): return None
        entries = self.memory.recall(key, top_k=1)
        try: return clamp_score(entries[0]["data"]["score"]) if entries else None
        except Exception: return None

    def _domain_age_cache_set(self, key: str, score: float) -> None:
        if not coerce_bool(self._cfg("domain_age.cache_enabled", True), True): return
        self.memory.add({"score": clamp_score(score), "timestamp": utc_iso()}, tags=[key, "adaptive_security", "domain_age"], sensitivity=0.3, ttl_seconds=coerce_int(self._cfg("domain_age.cache_ttl_seconds", 86400), 86400, minimum=60), purpose="adaptive_security_domain_age_cache", owner="adaptive_security", source="adaptive_security")

    def _extract_domain(self, url: str) -> str:
        try: return extract_domain(url)
        except Exception:
            return (urlparse(url).hostname or "").lower()

    # ------------------------------------------------------------------
    # Traffic, IP, and rate limiting
    # ------------------------------------------------------------------
    def _contains_ip_address(self, text: str, *, validate: bool = True, check_urls: bool = True, allow_private: bool = False, allow_reserved: bool = False) -> Union[bool, Tuple[bool, Optional[str]]]:
        candidates = self._IPV4_PATTERN.findall(text or "") + self._IPV6_PATTERN.findall(text or "")
        for value in candidates:
            if not validate: return True
            try:
                ip = ipaddress.ip_address(value)
                if ip.is_private and not allow_private: return True, "Private IP found"
                if ip.is_reserved and not allow_reserved: return True, "Reserved IP found"
                return True, "IP address found"
            except ValueError: continue
        return False if not validate else (False, "No valid IP found")

    def monitor_traffic(self, packet: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(packet, Mapping):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Traffic monitor requires a mapping packet.", component="adaptive_security")
        source_ip = normalize_text(packet.get("source_ip", "unknown"), max_length=128)
        anomalous = self._detect_anomalous_payload(packet)
        if anomalous: self._block_ip(source_ip, reason="anomalous_payload")
        with self._state_lock: blocked = source_ip in self.blocked_ips
        return sanitize_for_logging({"source_ip_fingerprint": fingerprint(source_ip), "anomalous": anomalous, "blocked": blocked, "timestamp": utc_iso()})

    def _detect_anomalous_payload(self, packet: Mapping[str, Any]) -> bool:
        payload = str(packet.get("payload", ""))
        return len(payload.encode("utf-8", errors="replace")) > coerce_int(self._cfg("traffic.payload_size_limit", self.input_size_limit), self.input_size_limit, minimum=1) or bool(re.search(r"[\x00-\x08\x0e-\x1f]", payload))

    def _block_ip(self, ip: str, *, reason: str = "policy") -> None:
        value = normalize_text(ip, max_length=128)
        with self._state_lock:
            self.blocked_ips[value] = {"reason": reason, "timestamp": utc_iso(), "fingerprint": fingerprint(value)}

    def _principal_from_context(self, client_ip: Optional[str], context: Optional[Mapping[str, Any]]) -> Optional[str]:
        context = context or {}
        raw = client_ip or context.get("client_ip") or context.get("principal") or context.get("session_id") or context.get("request_id")
        return normalize_identifier(raw, max_length=128) if raw else None

    def _rate_limit_decision(self, principal: Optional[str] = None) -> RateLimitDecision:
        if not principal:
            if coerce_bool(self._cfg("rate_limit_require_principal", False), False):
                return RateLimitDecision(False, "missing", 0, self.rate_limit, coerce_int(self._cfg("rate_limit_window_seconds", 60), 60), reason="missing_principal", applied=False)
            return RateLimitDecision(True, "unscoped", 0, self.rate_limit, coerce_int(self._cfg("rate_limit_window_seconds", 60), 60), reason="not_applied_missing_principal", applied=False)
        now = time.time(); window = coerce_int(self._cfg("rate_limit_window_seconds", 60), 60, minimum=1); limit = self.rate_limit
        with self._state_lock:
            queue = self.request_tracker[principal]
            while queue and now - queue[0] > window: queue.popleft()
            queue.append(now); count = len(queue); oldest = queue[0]
        if count > limit:
            return RateLimitDecision(False, principal, count, limit, window, max(0.0, window - (now - oldest)), "rate_limit_exceeded")
        return RateLimitDecision(True, principal, count, limit, window)

    def _check_input_overload(self) -> bool:
        return not self._rate_limit_decision(None).allowed

    # ------------------------------------------------------------------
    # Supply chain and training
    # ------------------------------------------------------------------
    def _load_trusted_hashes(self) -> Dict[str, str]:
        configured = self._cfg("trusted_hashes", {}) or {}
        if isinstance(configured, Mapping): return {str(k): str(v).lower() for k, v in configured.items() if v}
        if isinstance(configured, str) and configured:
            loaded = json.loads(load_text_file(configured, max_bytes=coerce_int(self._cfg("supply_chain.max_hash_file_bytes", 262144), 262144)))
            return {str(k): str(v).lower() for k, v in loaded.items()} if isinstance(loaded, Mapping) else {}
        return {}

    def check_supply_chain(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise SecurityError(SecurityErrorType.SUPPLY_CHAIN_COMPROMISE, "Supply-chain artifact is missing or not a file.", component="adaptive_security", context={"path_fingerprint": fingerprint(str(path))})
        maximum = coerce_int(self._cfg("supply_chain.max_file_bytes", 512_000_000), 512_000_000, minimum=1)
        if path.stat().st_size > maximum:
            raise ResourceExhaustionError("supply_chain_file_bytes", float(path.stat().st_size), float(maximum), source_identifier="adaptive_security", component="adaptive_security")
        digest = self._calculate_file_hash(str(path))
        matches = [name for name, trusted in self.safe_package_hashes.items() if hmac_compare_digest(str(trusted), digest)]
        trusted = bool(matches)
        block_unknown = coerce_bool(self._cfg("supply_chain.block_unknown", True), True)
        return SupplyChainCheckResult(fingerprint(str(path)), digest, trusted, "low" if trusted else "high" if block_unknown else "medium", "allow" if trusted else "block" if block_unknown else "review", matches[0] if matches else None, {"size_bytes": path.stat().st_size}).to_dict()

    def _calculate_file_hash(self, file_path: str) -> str:
        algorithm = safe_hash_algorithm(str(self._cfg("supply_chain.hash_algorithm", "sha256")))
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(coerce_int(self._cfg("supply_chain.chunk_size", 65536), 65536, minimum=4096)), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def train_phishing_model(self, model_type: str, training_data: List[Tuple[List[float], List[float]]], *, validation_data: Optional[List[Tuple[List[float], List[float]]]] = None, epochs: Optional[int] = None, batch_size: Optional[int] = None, learning_rate: Optional[float] = None) -> Dict[str, Any]:
        selected = self._select_model(model_type)
        model = selected["model"]
        if model is None:
            settings = self._model_settings(model_type)
            model = NeuralNetwork(settings["num_inputs"], settings["layer_config"], loss_function_name=settings["loss_function_name"], optimizer_name=settings["optimizer_name"], problem_type=settings["problem_type"], config=get_config_section("neural_network"))
        summary = model.train(training_data, validation_data=validation_data, epochs=epochs or coerce_int(self._cfg("training.epochs", 50), 50), batch_size=batch_size or coerce_int(self._cfg("training.batch_size", 32), 32), initial_learning_rate=learning_rate or coerce_float(self._cfg("training.learning_rate", 0.001), 0.001))
        path = selected["path"]
        if path:
            model.save_model(path)
        if normalize_identifier(model_type) == "email": self.email_nn = model
        else: self.url_nn = model
        return sanitize_for_logging({"model_type": model_type, "model_path_fingerprint": fingerprint(path), "samples": len(training_data), "saved": bool(path), "training_summary": to_jsonable(summary), "timestamp": utc_iso()})

    def _select_model(self, model_type: str) -> Dict[str, Any]:
        normalized = normalize_identifier(model_type, default="unknown")
        if normalized == "email": return {"model": self.email_nn, "path": self.email_model_path}
        if normalized == "url": return {"model": self.url_nn, "path": self.url_model_path}
        raise ConfigurationTamperingError("adaptive_security.model_type", f"Unsupported model type: {model_type}", component="adaptive_security")

    def _store_analysis_result(self, result: AdaptiveAnalysisResult) -> None:
        if not coerce_bool(self._cfg("memory.store_analysis", True), True): return
        self.memory.add(result.to_dict(), tags=list(self._cfg("memory.analysis_tags", ["adaptive_security", "analysis"])), sensitivity=coerce_float(self._cfg("memory.analysis_sensitivity", 0.65), 0.65), ttl_seconds=self._cfg("memory.analysis_ttl_seconds", 86400), purpose="adaptive_security_analysis", owner="adaptive_security", source="adaptive_security", metadata={"eligible_for_compliance": True, "analysis_schema": ANALYSIS_SCHEMA_VERSION})


def hmac_compare_digest(left: str, right: str) -> bool:
    import hmac
    return hmac.compare_digest(str(left).lower(), str(right).lower())


__all__ = [
    "MODULE_VERSION", 
    "ANALYSIS_SCHEMA_VERSION", 
    "SUPPLY_CHAIN_SCHEMA_VERSION", 
    "AdaptiveAnalysisResult", 
    "RateLimitDecision", 
    "SupplyChainCheckResult", 
    "AdaptiveSecurity"
]


if __name__ == "__main__":
    print("\n=== Running Adaptive Security ===\n")
    printer.status("TEST", "Adaptive Security initialized", "info")
    printer.section_header("Smoke test #1: Initialization")
    memory = SecureMemory()
    safety = SafetyFeatures()
    AdSe = AdaptiveSecurity(memory=memory, safety_features=safety)

    printer.status("START", "Adaptive Security ready", "success" if AdSe is not None else "error")

    printer.section_header("Smoke test #2: Public analysis")
    email = "Jean_thegoat2000@gmail.com"
    url = "https://www.google.com/webhp?hl=nl&sa=X&ved=2ahUKEwiz3NzUx_KWAxVJ2gIHHSUzAnsQPHoECAYQBA"

    analyze_email = AdSe.analyze_email(email={"from": email})
    analyze_url = AdSe.analyze_url(url=url)

    printer.status("EMAIL", analyze_email, "success" if analyze_email.get("decision") in {"allow", "review"} else "error")
    printer.status("URL",   analyze_url,   "success" if analyze_url.get("decision")   in {"allow", "review"} else "error")

    printer.section_header("Smoke test #3: Monitor")
    malicious_packet = {
    "source_ip": "198.51.100.7",
    "payload": "GET / HTTP/1.1\x00evil",
    }
    monitor = AdSe.monitor_traffic(packet=malicious_packet)
    printer.status("Monitor", monitor, "success" if monitor.get("decision") in {"allow", "review"} else "error")

    print("\n== Task run successfully ==\n")