"""
- PII/PHI/entity detection.
- Sensitive attribute tagging.
- Contextual sensitivity scoring.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils import (load_global_config, get_config_section,
                    PrivacyError, PrivacyErrorType, PrivacySeverity,
                    normalize_privacy_exception)
from .privacy_memory import PrivacyMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Data Identification")
printer = PrettyPrinter


@dataclass(frozen=True)
class EntityDetection:
    entity_type: str
    category: str
    path: str
    detector: str
    confidence: float
    severity: str
    preview: str
    matched_length: int
    field_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "category": self.category,
            "path": self.path,
            "detector": self.detector,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "preview": self.preview,
            "matched_length": self.matched_length,
            "field_name": self.field_name,
        }


@dataclass(frozen=True)
class SensitiveAttributeTag:
    attribute: str
    path: str
    reason: str
    confidence: float
    evidence_preview: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attribute": self.attribute,
            "path": self.path,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "evidence_preview": self.evidence_preview,
        }


class DataID:
    """Data Identification and Classification.

    The module performs entity detection across structured and semi-structured
    content, tags sensitive attributes, and computes a contextual sensitivity
    score that downstream privacy agents can use for runtime decisions.
    """
    def __init__(self) -> None:
        self.config = load_global_config()
        self.id_config = get_config_section("data_id")
        self._lock = RLock()

        self.defaults_template = self._load_defaults_template()
        self.entity_profiles = self._default_entity_profiles()

        self.enabled = bool(self.id_config.get("enabled", True))
        self.strict_mode = bool(self.id_config.get("strict_mode", True))
        self.sanitize_freeform_context = bool(self.id_config.get("sanitize_freeform_context", True))
        self.record_decisions_in_memory = bool(self.id_config.get("record_decisions_in_memory", True))
        self.write_shared_contract = bool(self.id_config.get("write_shared_contract", True))
        self.default_policy_version = str(self.id_config.get("default_policy_version", "v1"))
        self.default_decision_stage = str(self.id_config.get("default_decision_stage", "identification.runtime_gate"))
        self.default_text_stage = str(self.id_config.get("default_text_stage", "identification.text_scan"))

        self.max_depth = int(self.id_config.get("max_depth", self._runtime_default("max_depth", 12)))
        self.max_collection_items = int(
            self.id_config.get("max_collection_items", self._runtime_default("max_collection_items", 500))
        )
        self.max_matches = int(self.id_config.get("max_matches", self._runtime_default("max_matches", 500)))
        self.max_attribute_tags = int(
            self.id_config.get("max_attribute_tags", self._runtime_default("max_attribute_tags", 200))
        )
        self.max_context_preview = int(
            self.id_config.get("max_context_preview", self._runtime_default("max_context_preview", 48))
        )
        self.min_regex_confidence = float(
            self.id_config.get("min_regex_confidence", self._runtime_default("min_regex_confidence", 0.72))
        )
        self.min_field_confidence = float(
            self.id_config.get("min_field_confidence", self._runtime_default("min_field_confidence", 0.63))
        )
        self.min_keyword_confidence = float(
            self.id_config.get("min_keyword_confidence", self._runtime_default("min_keyword_confidence", 0.58))
        )
        self.escalation_uncertainty_threshold = float(
            self.id_config.get(
                "escalation_uncertainty_threshold",
                self._runtime_default("escalation_uncertainty_threshold", 0.22),
            )
        )
        self.medium_sensitivity_threshold = float(
            self.id_config.get(
                "medium_sensitivity_threshold",
                self._runtime_default("medium_sensitivity_threshold", 0.35),
            )
        )
        self.high_sensitivity_threshold = float(
            self.id_config.get(
                "high_sensitivity_threshold",
                self._runtime_default("high_sensitivity_threshold", 0.60),
            )
        )
        self.critical_sensitivity_threshold = float(
            self.id_config.get(
                "critical_sensitivity_threshold",
                self._runtime_default("critical_sensitivity_threshold", 0.85),
            )
        )

        self.regex_patterns = self._build_regex_patterns()
        self.field_keywords = self._build_field_keywords()
        self.attribute_keywords = self._build_attribute_keywords()
        self.entity_weights = self._build_entity_weights()
        self.severity_thresholds = self._build_severity_thresholds()

        self.memory = PrivacyMemory()
        self._shared_contract_fallback: Dict[str, Dict[str, Any]] = {}
        self._decision_trace_fallback: Dict[str, List[Dict[str, Any]]] = {}
        self._validate_config()

        logger.info("DataID initialized with production-ready detection and classification settings.")

    def _load_defaults_template(self) -> Dict[str, Any]:
        defaults_path_value = str(self.id_config.get("defaults_dir", "")).strip()
        if not defaults_path_value:
            raise PrivacyError(
                message="Missing 'defaults_dir' in data_id configuration.",
                error_type=PrivacyErrorType.DATA_CLASSIFICATION_POLICY_MISSING,
                severity=PrivacySeverity.HIGH,
                retryable=False,
                context={"config_section": "data_id", "field": "defaults_dir"},
                remediation="Set data_id.defaults_dir to the privacy defaults JSON template path.",
            )

        config_path = self.config.get("__config_path__")
        if config_path:
            config_root = Path(config_path).resolve().parents[1]
            defaults_path = config_root / defaults_path_value
        else:
            defaults_path = Path(defaults_path_value).resolve()

        if not defaults_path.exists():
            raise PrivacyError(
                message="Data identification defaults template could not be found.",
                error_type=PrivacyErrorType.DATA_CLASSIFICATION_POLICY_MISSING,
                severity=PrivacySeverity.HIGH,
                retryable=False,
                context={
                    "config_section": "data_id",
                    "defaults_dir": defaults_path_value,
                    "resolved_path": str(defaults_path),
                },
                remediation="Ensure privacy_defaults.json exists at the configured path.",
            )

        try:
            with open(defaults_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise PrivacyError(
                message="Failed to load data identification defaults template.",
                error_type=PrivacyErrorType.DATA_CLASSIFICATION_POLICY_MISSING,
                severity=PrivacySeverity.HIGH,
                retryable=False,
                context={
                    "config_section": "data_id",
                    "defaults_dir": defaults_path_value,
                    "resolved_path": str(defaults_path),
                    "details": str(exc),
                },
                remediation="Repair the defaults JSON file and verify valid JSON syntax.",
            ) from exc

        if not isinstance(payload, dict):
            raise PrivacyError(
                message="Data identification defaults template has an invalid structure.",
                error_type=PrivacyErrorType.DATA_CLASSIFICATION_POLICY_MISSING,
                severity=PrivacySeverity.HIGH,
                retryable=False,
                context={
                    "config_section": "data_id",
                    "defaults_dir": defaults_path_value,
                    "resolved_path": str(defaults_path),
                    "payload_type": type(payload).__name__,
                },
                remediation="Ensure privacy_defaults.json is a top-level JSON object.",
            )

        return payload

    def _runtime_default(self, key: str, fallback: Any) -> Any:
        runtime_defaults = self.defaults_template.get("runtime_defaults", {}) or {}
        return runtime_defaults.get(key, fallback)

    def _default_regex_patterns(self) -> Dict[str, str]:
        patterns: Dict[str, str] = {}
        raw_patterns = self.defaults_template.get("regex_patterns", {}) or {}
        for entity_type, spec in raw_patterns.items():
            if not isinstance(spec, Mapping):
                continue
            if spec.get("enabled", True) is False:
                continue
            pattern = spec.get("pattern")
            if pattern:
                patterns[str(entity_type)] = str(pattern)
        return patterns

    def _default_field_keywords(self) -> Dict[str, Tuple[str, str, float]]:
        keywords: Dict[str, Tuple[str, str, float]] = {}
        raw_rules = self.defaults_template.get("field_keyword_rules", {}) or {}
        for keyword, spec in raw_rules.items():
            if not isinstance(spec, Mapping):
                continue
            entity_type = str(spec.get("entity_type", "")).strip()
            category = str(spec.get("category", "contextual_identifier")).strip() or "contextual_identifier"
            confidence = float(spec.get("confidence", self.min_field_confidence))
            if entity_type:
                keywords[str(keyword)] = (entity_type, category, confidence)
        return keywords

    def _default_attribute_keywords(self) -> Dict[str, Dict[str, Any]]:
        defaults: Dict[str, Dict[str, Any]] = {}
        raw_keywords = self.defaults_template.get("sensitive_attribute_keywords", {}) or {}
        for attribute, spec in raw_keywords.items():
            if not isinstance(spec, Mapping):
                continue
            defaults[str(attribute)] = {
                "keywords": [str(item) for item in spec.get("keywords", []) if str(item).strip()],
                "confidence": float(spec.get("confidence", self.min_keyword_confidence)),
                "linked_entity_types": [str(item) for item in spec.get("linked_entity_types", []) if str(item).strip()],
                "regimes": [str(item) for item in spec.get("regimes", []) if str(item).strip()],
            }
        return defaults

    def _default_entity_profiles(self) -> Dict[str, Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}
        raw_profiles = self.defaults_template.get("entity_profiles", {}) or {}
        for entity_type, spec in raw_profiles.items():
            if not isinstance(spec, Mapping):
                continue
            profiles[str(entity_type)] = {
                "category": str(spec.get("category", "contextual_identifier")),
                "default_severity": str(spec.get("default_severity", PrivacySeverity.LOW.value)),
                "weight": float(spec.get("weight", 0.10)),
                "data_class": str(spec.get("data_class", "pii")),
                "recommended_controls": [
                    str(item) for item in spec.get("recommended_controls", []) if str(item).strip()
                ],
                "tags": [str(item) for item in spec.get("tags", []) if str(item).strip()],
            }
        return profiles

    def _default_severity_thresholds(self) -> Dict[str, float]:
        scoring = self.defaults_template.get("scoring", {}) or {}
        decision_thresholds = scoring.get("decision_thresholds", {}) or {}
        runtime_defaults = self.defaults_template.get("runtime_defaults", {}) or {}

        return {
            "low": float((self.id_config.get("severity_thresholds", {}) or {}).get("low", 0.15)),
            "medium": float(decision_thresholds.get("modify", runtime_defaults.get("medium_sensitivity_threshold", 0.35))),
            "high": float(decision_thresholds.get("high", runtime_defaults.get("high_sensitivity_threshold", 0.60))),
            "critical": float(decision_thresholds.get("critical", runtime_defaults.get("critical_sensitivity_threshold", 0.85))),
        }

    def _validate_config(self) -> None:
        positive_ints = {
            "max_depth": self.max_depth,
            "max_collection_items": self.max_collection_items,
            "max_matches": self.max_matches,
            "max_attribute_tags": self.max_attribute_tags,
            "max_context_preview": self.max_context_preview,
        }
        for field_name, value in positive_ints.items():
            if value <= 0:
                raise PrivacyError(
                    message=f"Invalid data_id configuration for '{field_name}'.",
                    error_type=PrivacyErrorType.PII_DETECTION_FAILED,
                    severity=PrivacySeverity.HIGH,
                    retryable=False,
                    context={"field_name": field_name, "value": value},
                    remediation="Correct the data_id configuration before running classification.",
                )

        bounded_values = {
            "min_regex_confidence": self.min_regex_confidence,
            "min_field_confidence": self.min_field_confidence,
            "min_keyword_confidence": self.min_keyword_confidence,
            "escalation_uncertainty_threshold": self.escalation_uncertainty_threshold,
            "medium_sensitivity_threshold": self.medium_sensitivity_threshold,
            "high_sensitivity_threshold": self.high_sensitivity_threshold,
            "critical_sensitivity_threshold": self.critical_sensitivity_threshold,
        }
        for field_name, value in bounded_values.items():
            if not 0.0 <= value <= 1.0:
                raise PrivacyError(
                    message=f"Invalid data_id configuration for '{field_name}'.",
                    error_type=PrivacyErrorType.PII_DETECTION_FAILED,
                    severity=PrivacySeverity.HIGH,
                    retryable=False,
                    context={"field_name": field_name, "value": value},
                    remediation="Use normalized probability-style thresholds between 0.0 and 1.0.",
                )

        if not (self.medium_sensitivity_threshold <= self.high_sensitivity_threshold <= self.critical_sensitivity_threshold):
            raise PrivacyError(
                message="Sensitivity thresholds are not monotonically increasing.",
                error_type=PrivacyErrorType.PII_DETECTION_FAILED,
                severity=PrivacySeverity.HIGH,
                retryable=False,
                context={
                    "medium": self.medium_sensitivity_threshold,
                    "high": self.high_sensitivity_threshold,
                    "critical": self.critical_sensitivity_threshold,
                },
                remediation="Ensure medium <= high <= critical for sensitivity scoring.",
            )

    def _build_regex_patterns(self) -> Dict[str, re.Pattern[str]]:
        raw_patterns = self._default_regex_patterns()
        raw_patterns.update(self.id_config.get("regex_patterns", {}) or {})

        compiled: Dict[str, re.Pattern[str]] = {}
        for name, pattern in raw_patterns.items():
            try:
                compiled[str(name)] = re.compile(str(pattern), re.IGNORECASE)
            except re.error as exc:
                raise PrivacyError(
                    message=f"Invalid regex pattern configured for entity '{name}'.",
                    error_type=PrivacyErrorType.DATA_CLASSIFICATION_POLICY_MISSING,
                    severity=PrivacySeverity.HIGH,
                    retryable=False,
                    context={"entity_type": name, "pattern": str(pattern), "details": str(exc)},
                    remediation="Fix the regex definition in privacy_defaults.json or config overrides.",
                ) from exc
        return compiled

    def _build_field_keywords(self) -> Dict[str, Tuple[str, str, float]]:
        keywords = dict(self._default_field_keywords())
        overrides = self.id_config.get("field_keywords", {}) or {}

        for key, value in overrides.items():
            if isinstance(value, Mapping):
                entity_type = str(value.get("entity_type", "")).strip()
                category = str(value.get("category", "contextual_identifier")).strip() or "contextual_identifier"
                confidence = float(value.get("confidence", self.min_field_confidence))
                if entity_type:
                    keywords[str(key)] = (entity_type, category, confidence)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
                keywords[str(key)] = (
                    str(value[0]),
                    str(value[1]),
                    self.min_field_confidence,
                )
            else:
                keywords[str(key)] = (str(value), "contextual_identifier", self.min_field_confidence)

        return keywords

    def _build_attribute_keywords(self) -> Dict[str, Dict[str, Any]]:
        keywords = dict(self._default_attribute_keywords())
        overrides = self.id_config.get("attribute_keywords", {}) or {}

        for key, value in overrides.items():
            if isinstance(value, Mapping):
                keywords[str(key)] = {
                    "keywords": [str(item) for item in value.get("keywords", []) if str(item).strip()],
                    "confidence": float(value.get("confidence", self.min_keyword_confidence)),
                    "linked_entity_types": [
                        str(item) for item in value.get("linked_entity_types", []) if str(item).strip()
                    ],
                    "regimes": [str(item) for item in value.get("regimes", []) if str(item).strip()],
                }
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                keywords[str(key)] = {
                    "keywords": [str(item) for item in value if str(item).strip()],
                    "confidence": self.min_keyword_confidence,
                    "linked_entity_types": [],
                    "regimes": [],
                }

        return keywords

    def _build_entity_weights(self) -> Dict[str, float]:
        weights = {
            entity_type: float(profile.get("weight", 0.10))
            for entity_type, profile in self.entity_profiles.items()
        }
        overrides = self.id_config.get("entity_weights", {}) or {}
        for key, value in overrides.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return weights

    def _build_severity_thresholds(self) -> Dict[str, float]:
        defaults = self._default_severity_thresholds()
        overrides = self.id_config.get("severity_thresholds", {}) or {}

        thresholds = {
            "low": float(overrides.get("low", defaults["low"])),
            "medium": float(overrides.get("medium", self.medium_sensitivity_threshold)),
            "high": float(overrides.get("high", self.high_sensitivity_threshold)),
            "critical": float(overrides.get("critical", self.critical_sensitivity_threshold)),
        }
        return thresholds

    def _severity_threshold_value(self, key: str, fallback: float) -> float:
        thresholds = self.id_config.get("severity_thresholds", {}) or {}
        try:
            value = float(thresholds.get(key, fallback))
        except (TypeError, ValueError):
            value = fallback
        return value

    @staticmethod
    def _normalize_identity(value: str, field_name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"'{field_name}' must be a non-empty string")
        return normalized

    def _normalize_context(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not value:
            return {}
        if not self.sanitize_freeform_context:
            return dict(value)

        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if any(token in key_str.lower() for token in ("payload", "content", "text", "body", "prompt")):
                sanitized[key_str] = self._preview_text(item)
            else:
                sanitized[key_str] = item
        return sanitized

    def _preview_text(self, value: Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.max_context_preview:
            return text
        return f"{text[: self.max_context_preview - 3]}..."

    def _mask_match(self, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return ""
        if len(stripped) <= 4:
            return "*" * len(stripped)
        return f"{stripped[:2]}***{stripped[-2:]}"

    def _iter_nodes(self, payload: Any, *, path: str = "root", depth: int = 0) -> Iterable[Tuple[str, Optional[str], Any]]:
        if depth > self.max_depth:
            return

        if isinstance(payload, Mapping):
            items = list(payload.items())[: self.max_collection_items]
            for key, value in items:
                key_str = str(key)
                child_path = f"{path}.{key_str}"
                yield from self._iter_nodes(value, path=child_path, depth=depth + 1)
            return

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            values = list(payload)[: self.max_collection_items]
            for index, value in enumerate(values):
                child_path = f"{path}[{index}]"
                yield from self._iter_nodes(value, path=child_path, depth=depth + 1)
            return

        field_name = path.rsplit(".", 1)[-1] if "." in path else path
        yield path, field_name, payload

    def _field_tokens(self, field_name: Optional[str]) -> List[str]:
        if not field_name:
            return []
        normalized_field = re.sub(r"[^a-z0-9_]+", "_", field_name.lower())
        return [token for token in normalized_field.split("_") if token]

    def _should_skip_field_heuristic(self, entity_type: str, keyword: str, field_name: Optional[str]) -> bool:
        tokens = set(self._field_tokens(field_name))
        if entity_type == "address" and "ip" in tokens:
            return True
        if entity_type == "phone" and {"account", "id"}.intersection(tokens) and not {"phone", "mobile"}.intersection(tokens):
            return True
        if entity_type == "email" and "filename" in tokens:
            return True
        return False

    def _should_skip_regex_detection(self, entity_type: str, field_name: Optional[str], matched_text: str) -> bool:
        tokens = set(self._field_tokens(field_name))
        normalized_match = matched_text.strip()

        if entity_type == "phone" and {"account", "id"}.intersection(tokens) and not {"phone", "mobile"}.intersection(tokens):
            return True
        if entity_type == "address" and "ip" in tokens:
            return True
        if entity_type == "national_id":
            if not {"id", "national", "government", "tax", "passport", "ssn"}.intersection(tokens):
                return True
        if entity_type == "credit_card":
            digits = re.sub(r"\D+", "", normalized_match)
            if len(digits) < 13:
                return True
        return False

    def _detect_from_field_name(self, field_name: Optional[str], path: str, value: Any) -> List[EntityDetection]:
        detections: List[EntityDetection] = []
        if not field_name:
            return detections

        normalized_field = re.sub(r"[^a-z0-9_]+", "_", field_name.lower())
        text_preview = self._preview_text(value)

        for keyword, (entity_type, category, confidence) in self.field_keywords.items():
            if keyword in normalized_field:
                if self._should_skip_field_heuristic(entity_type, keyword, field_name):
                    continue
                detections.append(
                    EntityDetection(
                        entity_type=entity_type,
                        category=category,
                        path=path,
                        detector="field_heuristic",
                        confidence=confidence,
                        severity=self._entity_severity(entity_type),
                        preview=text_preview,
                        matched_length=len(str(value)) if value is not None else 0,
                        field_name=field_name,
                    )
                )
        return detections

    def _detect_from_regex(self, field_name: Optional[str], path: str, value: Any) -> List[EntityDetection]:
        if value is None:
            return []

        text = value if isinstance(value, str) else str(value)
        if not text.strip():
            return []

        detections: List[EntityDetection] = []
        for entity_type, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                if self._should_skip_regex_detection(entity_type, field_name, matched_text):
                    continue
                detections.append(
                    EntityDetection(
                        entity_type=entity_type,
                        category=self._category_for_entity(entity_type),
                        path=path,
                        detector="regex",
                        confidence=self._regex_confidence(entity_type),
                        severity=self._entity_severity(entity_type),
                        preview=self._mask_match(matched_text),
                        matched_length=len(matched_text),
                        field_name=field_name,
                    )
                )
                if len(detections) >= self.max_matches:
                    return detections
        return detections

    def _detect_sensitive_attributes(self, field_name: Optional[str], path: str, value: Any) -> List[SensitiveAttributeTag]:
        if value is None:
            return []

        text = value if isinstance(value, str) else str(value)
        combined = f"{field_name or ''} {text}".lower()
        tags: List[SensitiveAttributeTag] = []

        for attribute, spec in self.attribute_keywords.items():
            keywords = spec.get("keywords", [])
            confidence = float(spec.get("confidence", self.min_keyword_confidence))

            for keyword in keywords:
                if re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", combined):
                    tags.append(
                        SensitiveAttributeTag(
                            attribute=attribute,
                            path=path,
                            reason=f"keyword_match:{keyword}",
                            confidence=confidence,
                            evidence_preview=self._preview_text(text),
                        )
                    )
                    break

            if len(tags) >= self.max_attribute_tags:
                break

        return tags

    def _category_for_entity(self, entity_type: str) -> str:
        if entity_type in {"email", "phone", "address", "date_of_birth"}:
            return "direct_identifier"
        if entity_type in {"ssn", "passport", "national_id"}:
            return "government_identifier"
        if entity_type in {"credit_card", "account_id"}:
            return "financial_identifier"
        if entity_type in {"patient", "diagnosis", "medical_record"}:
            return "health_attribute"
        if entity_type in {"session_token", "access_token", "password", "secret"}:
            return "credential"
        return "contextual_identifier"

    def _entity_severity(self, entity_type: str) -> str:
        profile = self.entity_profiles.get(entity_type, {})
        explicit_severity = str(profile.get("default_severity", "")).strip().lower()
        if explicit_severity in {
            PrivacySeverity.LOW.value,
            PrivacySeverity.MEDIUM.value,
            PrivacySeverity.HIGH.value,
            PrivacySeverity.CRITICAL.value,
        }:
            return explicit_severity

        weight = self.entity_weights.get(entity_type, 0.12)
        if weight >= self.critical_sensitivity_threshold:
            return PrivacySeverity.CRITICAL.value
        if weight >= self.high_sensitivity_threshold:
            return PrivacySeverity.HIGH.value
        if weight >= self.medium_sensitivity_threshold:
            return PrivacySeverity.MEDIUM.value
        return PrivacySeverity.LOW.value

    def _deduplicate_entities(self, detections: Sequence[EntityDetection]) -> List[EntityDetection]:
        deduped: Dict[Tuple[str, str, str, str], EntityDetection] = {}
        for detection in detections:
            key = (detection.entity_type, detection.path, detection.preview, detection.detector)
            current = deduped.get(key)
            if current is None or detection.confidence > current.confidence:
                deduped[key] = detection
        result = list(deduped.values())
        result.sort(key=lambda item: (-item.confidence, item.entity_type, item.path))
        return result[: self.max_matches]

    def _deduplicate_tags(self, tags: Sequence[SensitiveAttributeTag]) -> List[SensitiveAttributeTag]:
        deduped: Dict[Tuple[str, str, str], SensitiveAttributeTag] = {}
        for tag in tags:
            key = (tag.attribute, tag.path, tag.reason)
            current = deduped.get(key)
            if current is None or tag.confidence > current.confidence:
                deduped[key] = tag
        result = list(deduped.values())
        result.sort(key=lambda item: (-item.confidence, item.attribute, item.path))
        return result[: self.max_attribute_tags]

    def _compute_sensitivity_score(self, detections: Sequence[EntityDetection],
                                   sensitive_attributes: Sequence[SensitiveAttributeTag], *,
                                   context: Optional[Mapping[str, Any]] = None) -> float:
        score = 0.0
        for detection in detections:
            score += self.entity_weights.get(detection.entity_type, 0.10) * detection.confidence

        scoring = self.defaults_template.get("scoring", {}) or {}
        attribute_bonus_per_match = float(scoring.get("attribute_tag_bonus_per_match", 0.06))
        attribute_bonus_cap = float(scoring.get("attribute_bonus_cap", 0.24))

        if sensitive_attributes:
            score += min(attribute_bonus_cap, attribute_bonus_per_match * len(sensitive_attributes))

        normalized_context = self._normalize_context(context)
        context_adjustments = scoring.get("context_adjustments", {}) or {}
        source_context_adjustments = context_adjustments.get("source_context", {}) or {}
        purpose_adjustments = context_adjustments.get("purpose", {}) or {}

        source_context = normalized_context.get("source_context")
        purpose = normalized_context.get("purpose")

        if source_context in source_context_adjustments:
            score += float(source_context_adjustments[source_context])

        if purpose in purpose_adjustments:
            score += float(purpose_adjustments[purpose])

        return round(min(score, 1.0), 4)

    def _derive_decision(self, score: float, uncertain_fraction: float) -> str:
        if uncertain_fraction >= self.escalation_uncertainty_threshold:
            return "escalate"
        if score >= self.medium_sensitivity_threshold:
            return "modify"
        return "allow"

    def _derive_severity(self, score: float) -> str:
        if score >= self.critical_sensitivity_threshold:
            return PrivacySeverity.CRITICAL.value
        if score >= self.high_sensitivity_threshold:
            return PrivacySeverity.HIGH.value
        if score >= self.medium_sensitivity_threshold:
            return PrivacySeverity.MEDIUM.value
        return PrivacySeverity.LOW.value

    def _build_recommended_actions(
        self,
        detections: Sequence[EntityDetection],
        sensitive_attributes: Sequence[SensitiveAttributeTag],
        decision: str,
    ) -> List[str]:
        rules = self.defaults_template.get("recommendation_rules", {}) or {}

        default_entity_actions = [
            str(item) for item in rules.get("default_actions_if_entities_present", []) if str(item).strip()
        ]
        default_attribute_actions = [
            str(item) for item in rules.get("default_actions_if_attributes_present", []) if str(item).strip()
        ]
        high_risk_entity_types = {
            str(item) for item in rules.get("high_risk_entity_types", []) if str(item).strip()
        }
        high_risk_actions = [
            str(item) for item in rules.get("high_risk_actions", []) if str(item).strip()
        ]
        decision_actions = rules.get("decision_actions", {}) or {}

        actions: List[str] = []

        if detections:
            actions.extend(default_entity_actions)
        if any(d.entity_type in high_risk_entity_types for d in detections):
            actions.extend(high_risk_actions)
        if sensitive_attributes:
            actions.extend(default_attribute_actions)
        actions.extend([str(item) for item in decision_actions.get(decision, []) if str(item).strip()])

        deduped: List[str] = []
        seen = set()
        for action in actions:
            if action not in seen:
                deduped.append(action)
                seen.add(action)

        return deduped

    def _memory_supports(self, method_name: str) -> bool:
        return hasattr(self.memory, method_name) and callable(getattr(self.memory, method_name))

    def _write_memory_contract(
        self,
        *,
        request_id: str,
        sensitivity_score: float,
        detected_entities: Sequence[Mapping[str, Any]],
        audit_trail_ref: Optional[str],
    ) -> None:
        payload = {
            "request_id": request_id,
            "privacy.sensitivity_score": sensitivity_score,
            "privacy.detected_entities": list(detected_entities),
            "privacy.audit_trail_ref": audit_trail_ref,
            "last_updated_at": time.time(),
        }
        if self._memory_supports("write_request_contract"):
            self.memory.write_request_contract(
                request_id=request_id,
                sensitivity_score=sensitivity_score,
                detected_entities=list(detected_entities),
                audit_trail_ref=audit_trail_ref,
            )
            return

        self._shared_contract_fallback[request_id] = payload
        logger.warning("PrivacyMemory does not expose 'write_request_contract'; using DataID compatibility cache.")

    def _record_memory_decision(
        self,
        *,
        request_id: str,
        stage: str,
        decision: str,
        rationale: str,
        subject_id: Optional[str],
        record_id: Optional[str],
        purpose: Optional[str],
        sensitivity_score: float,
        detected_entities: Sequence[Mapping[str, Any]],
        policy_id: Optional[str],
        policy_version: str,
        context: Mapping[str, Any],
        audit_trail_ref: Optional[str],
    ) -> None:
        event = {
            "request_id": request_id,
            "stage": stage,
            "decision": decision,
            "rationale": rationale,
            "subject_id": subject_id,
            "record_id": record_id,
            "purpose": purpose,
            "sensitivity_score": sensitivity_score,
            "detected_entities": list(detected_entities),
            "policy_id": policy_id,
            "policy_version": policy_version,
            "context": dict(context),
            "audit_trail_ref": audit_trail_ref,
            "timestamp": time.time(),
        }
        if self._memory_supports("record_privacy_decision"):
            self.memory.record_privacy_decision(
                request_id=request_id,
                stage=stage,
                decision=decision,
                rationale=rationale,
                subject_id=subject_id,
                record_id=record_id,
                purpose=purpose,
                sensitivity_score=sensitivity_score,
                detected_entities=list(detected_entities),
                policy_id=policy_id,
                policy_version=policy_version,
                context=dict(context),
                audit_trail_ref=audit_trail_ref,
            )
            return

        self._decision_trace_fallback.setdefault(request_id, []).append(event)
        logger.warning("PrivacyMemory does not expose 'record_privacy_decision'; using DataID compatibility cache.")

    def identify_entities(
        self,
        payload: Any,
        *,
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        source_context: Optional[str] = None,
        purpose: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        stage: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "identify_entities"
        try:
            if not self.enabled:
                raise PrivacyError(
                    message="data_id is disabled by configuration.",
                    error_type=PrivacyErrorType.PII_DETECTION_FAILED,
                    severity=PrivacySeverity.HIGH,
                    retryable=False,
                    context={"config_section": "data_id", "operation": operation},
                    remediation="Enable the data_id module before performing classification.",
                )

            normalized_context = self._normalize_context(context)
            effective_stage = stage or self.default_decision_stage
            effective_policy_version = str(policy_version or self.default_policy_version)

            all_entities: List[EntityDetection] = []
            all_tags: List[SensitiveAttributeTag] = []
            scanned_nodes = 0
            for path, field_name, value in self._iter_nodes(payload):
                scanned_nodes += 1
                all_entities.extend(self._detect_from_field_name(field_name, path, value))
                all_entities.extend(self._detect_from_regex(field_name, path, value))
                all_tags.extend(self._detect_sensitive_attributes(field_name, path, value))
                if len(all_entities) >= self.max_matches and len(all_tags) >= self.max_attribute_tags:
                    break

            deduped_entities = self._deduplicate_entities(all_entities)
            deduped_tags = self._deduplicate_tags(all_tags)
            score = self._compute_sensitivity_score(
                deduped_entities,
                deduped_tags,
                context={**normalized_context, "source_context": source_context, "purpose": purpose},
            )

            uncertain_entities = [item for item in deduped_entities if item.confidence < self.min_regex_confidence]
            uncertain_fraction = 0.0
            if deduped_entities:
                uncertain_fraction = len(uncertain_entities) / float(len(deduped_entities))

            decision = self._derive_decision(score, uncertain_fraction)
            severity = self._derive_severity(score)
            recommended_actions = self._build_recommended_actions(deduped_entities, deduped_tags, decision)
            entity_counts: Dict[str, int] = {}
            for item in deduped_entities:
                entity_counts[item.entity_type] = entity_counts.get(item.entity_type, 0) + 1

            result = {
                "request_id": request_id,
                "record_id": record_id,
                "subject_id": subject_id,
                "stage": effective_stage,
                "policy_id": policy_id,
                "policy_version": effective_policy_version,
                "decision": decision,
                "severity": severity,
                "sensitivity_score": score,
                "uncertainty_score": round(uncertain_fraction, 4),
                "entity_count": len(deduped_entities),
                "attribute_tag_count": len(deduped_tags),
                "scanned_nodes": scanned_nodes,
                "detected_entities": [item.to_dict() for item in deduped_entities],
                "sensitive_attributes": [item.to_dict() for item in deduped_tags],
                "entity_counts": entity_counts,
                "recommended_actions": recommended_actions,
                "audit_trail_ref": audit_trail_ref,
            }

            safe_contract_entities = [
                {
                    "entity_type": item.entity_type,
                    "category": item.category,
                    "path": item.path,
                    "confidence": round(item.confidence, 4),
                    "severity": item.severity,
                }
                for item in deduped_entities
            ]

            if request_id and self.write_shared_contract:
                self._write_memory_contract(
                    request_id=request_id,
                    sensitivity_score=score,
                    detected_entities=safe_contract_entities,
                    audit_trail_ref=audit_trail_ref,
                )

            if request_id and self.record_decisions_in_memory:
                self._record_memory_decision(
                    request_id=request_id,
                    stage=effective_stage,
                    decision=decision,
                    rationale=(
                        "Entity detection and contextual sensitivity scoring completed for the request. "
                        f"Detected {len(deduped_entities)} entity matches and {len(deduped_tags)} sensitive attribute tags."
                    ),
                    subject_id=subject_id,
                    record_id=record_id,
                    purpose=purpose,
                    sensitivity_score=score,
                    detected_entities=safe_contract_entities,
                    policy_id=policy_id,
                    policy_version=effective_policy_version,
                    context={
                        "source_context": source_context,
                        "entity_counts": entity_counts,
                        "recommended_actions": recommended_actions,
                        **normalized_context,
                    },
                    audit_trail_ref=audit_trail_ref,
                )

            return result
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_id.identify_entities",
                context={
                    "request_id": request_id,
                    "record_id": record_id,
                    "subject_id": subject_id,
                    "source_context": source_context,
                    "purpose": purpose,
                },
            ) from exc

    def _regex_confidence(self, entity_type: str) -> float:
        raw_patterns = self.defaults_template.get("regex_patterns", {}) or {}
        spec = raw_patterns.get(entity_type, {})
        if isinstance(spec, Mapping):
            try:
                return float(spec.get("confidence", self.min_regex_confidence))
            except (TypeError, ValueError):
                pass
        return self.min_regex_confidence

    def classify_text(self, text: str, *,
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        source_context: Optional[str] = None,
        purpose: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_text = self._normalize_identity(text, "text")
        return self.identify_entities(
            {"text": normalized_text},
            request_id=request_id,
            record_id=record_id,
            subject_id=subject_id,
            source_context=source_context,
            purpose=purpose,
            policy_id=policy_id,
            policy_version=policy_version,
            stage=self.default_text_stage,
            audit_trail_ref=audit_trail_ref,
            context=context,
        )

    def entity_summary(self, payload: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.identify_entities(payload, **kwargs)
        return {
            "decision": result["decision"],
            "severity": result["severity"],
            "sensitivity_score": result["sensitivity_score"],
            "entity_counts": result["entity_counts"],
            "recommended_actions": result["recommended_actions"],
        }

    def _handle_exception(self, exc: Exception, *, stage: str,
                          context: Optional[Dict[str, Any]] = None) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(exc, stage=stage, context=context)


if __name__ == "__main__":
    print("\n=== Running data ID===\n")
    printer.status("TEST", "data ID initialized", "info")

    classifier = DataID()

    sample_payload = {
        "ticket_id": "case-001",
        "customer": {
            "full_name": "Jane Doe",
            "email": "jane.doe@example.com",
            "phone": "+1 202-555-0188",
            "dob": "1992-07-14",
            "address": "1200 Example Avenue, Amsterdam",
            "account_id": "ACC-8827139",
        },
        "medical_note": "Patient diagnosis indicates follow-up treatment and allergy review.",
        "session": {
            "ip_address": "192.168.10.25",
            "authorization": "Bearer token-example-value",
        },
        "notes": [
            "Please do not store the raw passport value P12345678 in downstream tools.",
            "Secondary contact email is alt.user@sample.org",
        ],
    }

    printer.status("TEST", "Running structured payload identification", "info")
    payload_result = classifier.identify_entities(
        sample_payload,
        request_id="req-data-id-001",
        record_id="record-data-id-001",
        subject_id="subject-data-id-001",
        source_context="support_workspace",
        purpose="support_resolution",
        policy_id="data-identification-policy",
        policy_version="2026.04",
        context={"channel": "chat", "payload": sample_payload},
    )

    printer.status("TEST", "Running text classification", "info")
    text_result = classifier.classify_text(
        "Contact patient via john.smith@hospital.example and +31 6 12345678. Diagnosis remains confidential.",
        request_id="req-data-id-002",
        record_id="record-data-id-002",
        subject_id="subject-data-id-002",
        source_context="medical",
        purpose="diagnostics",
        policy_id="data-identification-policy",
        policy_version="2026.04",
    )

    printer.status("TEST", "Building summary", "info")
    summary = classifier.entity_summary(
        sample_payload,
        request_id="req-data-id-003",
        record_id="record-data-id-003",
        subject_id="subject-data-id-003",
        source_context="support_workspace",
        purpose="case_triage",
        policy_id="data-identification-policy",
        policy_version="2026.04",
    )

    print(json.dumps({
        "payload_result": payload_result,
        "text_result": text_result,
        "summary": summary,
    }, indent=2, default=str))

    print("\n=== Test ran successfully ===\n")
